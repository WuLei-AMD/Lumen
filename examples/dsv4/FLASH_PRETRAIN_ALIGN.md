# Lumen vs Miles DSV4 Flash native pretrain 对齐与反向排查

当天排查纪要（按发现顺序）：[FLASH_PRETRAIN_DEBUG_20260827.md](./FLASH_PRETRAIN_DEBUG_20260827.md)

本文记录截至 **2026-08-27** 在 2×8 MI308X 上的比对结论，以及在其他机器上复现的步骤。

代码分支（本地）：

| Repo | Branch | 说明 |
|------|--------|------|
| Lumen | `dsv4_mi325` | dump / TileLang 强制开关 / kernel compare 脚本 |
| Miles | `dsv4_mi325` | native pretrain、alignment dump、**TileLang fwd + Triton bwd** |

参考配置：TP4 / PP4 / EP4，Flash 43L，GBS=8，SEQ=2048，lr=1e-6，optimizer offload 0.6，**1 step**。

---

## 1. 结论（先看这个）

### 1.1 前向：已对齐

- **lm loss ≈ 12.84**（Lumen Megatron log 与 Miles `train/lm loss`）。
- 已对齐并固化的差异：
  1. next-token 切分：Miles 设 `MILES_DSV4_LUMEN_MOCK=1`（N token + extra label）。
  2. CE：Miles 走 fused vocab-parallel CE + `MILES_DSV4_LUMEN_LOSS_SCALE`。
  3. 同一 ckpt；freeze gate / e-score；hash 0–2 `tid2eid` / `expert_bias` skip。
- 不是 LN / wkv / compressor 前向问题。last-PP `after_mla_o` / `after_attn_out` / attn IO（q/kv/o/topk）约 **1.00×**。

### 1.2 反向：grad_norm 未完全对齐；主因是 sparse MLA 的 `dkv`

默认配置（Lumen **Triton** MLA，Miles **TileLang** MLA）：

| | lm loss | grad_norm |
|--|---------|-----------|
| Miles TileLang | ~12.84 | **~31** |
| Lumen Triton（默认） | ~12.84 | **~16** |
| Lumen `DSV4_FORCE_MLA_BACKEND=tilelang` | ~12.84 | **~35** |

- `V4_SPARSE_MLA_BACKEND=tilelang` **不会**改 Lumen kernel。Lumen 只认 `DSV4_FORCE_MLA_BACKEND`。
- 已排除：KV TP 缺 allreduce、fp32 vs bf16 allreduce、hash routing、CE reduction、transformer LN、compressor 实现、MHC 全局 2×。
- last-PP dump（dO 对齐后）：
  - Triton：`dkv_mla_in` Miles/Lumen ≈ **1.63×**，随后 LN/残差放大。
  - 两边都 TileLang：`dkv_*` / `after_wkv` ≈ **1.00×**，`mla_dq` ≈ **0.95×**。
- **单侧 kernel compare**（随机输入，同 shape）：
  - `o` / `dq` / `dsink`：cosine ≈ 1。
  - **`dkv`：能量 ≈ 1.0，cosine ≈ 0.39**。
  - vs FP32 ref：**Triton `dkv` 对齐 ref；TileLang `dkv` 对不上**。
- `dkv` shape 与 `kv` 相同，MQA 无 head 维。last-PP compress 层 MLA 入口：**`(1, 2560, 512)`**（2048 vanilla + 512 compress）。`dq` 是 `(1, 2048, 16, 512)`。

### 1.3 两个 bug 修完后的 grad_norm

两边都用 **Triton** sparse MLA bwd（Miles 见 1.4，Lumen 为默认）：

| 配置 | lm loss | grad_norm |
|------|---------|-----------|
| Miles TileLang bwd（修前） | 12.8385 | **31.09** |
| Miles Triton bwd（修后） | 12.8385 | **17.13** |
| Lumen 默认（缺 dgrad allreduce） | 12.8367 | **16.40** |
| Lumen + dgrad allreduce 修复 | 12.8370 | **19.09** |

修完 sparse MLA `dkv` 后 Miles 从 31.09 掉到 17.13，说明 2× 缺口主因确实是 TileLang `dkv`。

### 1.4 Lumen 代码修复：`LumenColumnParallelLinear` 缺 TP dgrad allreduce

`lumen/modules/parallel_linear.py`：

Megatron 原生 `ColumnParallelLinear` 在 `allreduce_dgrad=True` 时不走
`copy_to_tensor_model_parallel_region`，而是把 `allreduce_dgrad` 传进
`linear_with_grad_accumulation_and_async_allreduce`，由 GEMM 的 autograd 在 bwd 里
allreduce dgrad。Lumen 照抄了前半段分支，但 `_do_gemm` 没有 `allreduce_dgrad` 参数，
**这个 allreduce 谁都没做**。

- 触发条件：`tp_size > 1` 且 `sequence_parallel=False`。正常 transformer 全程开 SP，
  所以只有 DSV4 attention 里用 `config_no_sp` 建的 `wq_b` / `wo_a` 受影响。
- 现象：last-PP `after_wq_a` / `dx_wq` = Lumen/Miles **0.24×**（≈ 1/TP）。
- 修法：`allreduce_dgrad` 时改回走 `copy_to_tensor_model_parallel_region`
  （fwd identity / bwd allreduce，语义等价，只是同步而非异步 overlap）。
- 对照 Miles：`te_or_local.py` 在 gfx942 自动禁 TE，Miles 走的是 Megatron 原生
  `ColumnParallelLinear`，dgrad allreduce 正确。

修复后全 43 层 tp0 激活梯度比值（Lumen/Miles）：

| site | 修前 | 修后 |
|------|------|------|
| `after_wq_a` | 0.09 – 0.39 | 0.83 – 1.06 |
| `dx_wq` | 0.265 | 0.999 |
| `dkv_mla_in` / `after_mla_o` / `after_input_ln` | 0.63 – 1.14 | 0.87 – 1.06 |

唯一残留 outlier 是 **L22**（PP1 最后一层）全站点 ≈ 0.64，疑似 PP 边界效应。

`launch_dsv4_flash_pretrain_2node.sh` 的 rsync 列表要加
`lumen/modules/parallel_linear.py`，否则 worker 上的改动不生效。

### 1.5 Lumen 另两个缺陷：grad_norm 记账 + SP layernorm 归约

`DSV4_DUMP_GRAD_NORM=1` 出 per-param `gsq`，`sqrt(sum(gsq[inc]))` 精确复现 Megatron 的
`grad norm`（Lumen 19.0830 vs 日志 19.083），所以可以逐参数定位。用
`tools/compare_grad_norm_dump.py` 对比：

| bucket | Lumen sq | Miles sq | gn ratio |
|--------|----------|----------|----------|
| attn | 212.30 | 126.57 | 1.295 |
| shared_expert | 97.07 | 96.19 | 1.005 |
| compressor | 30.32 | 29.61 | 1.012 |
| other | 17.99 | 18.32 | 0.991 |
| **norm** | **3.84** | **16.71** | **0.479** |
| expert | 2.59 | 2.55 | 1.007 |

**缺陷 A：duplicated 权重被标成 tensor-parallel。**
`lumen/models/dsv4/megatron/layers.py` 的 `LumenDuplicatedLinear.__init__` 调了
`set_tensor_model_parallel_attributes(self.weight, True, 0, 1)`。这个权重在每个 TP rank
上是**完全复制**的，标成 TP-parallel 后 Megatron 的 grad_norm 在 4 个 rank 上各数一遍。

- 证据：`wq_a.weight` / `wkv.weight` 在 Lumen dump 里 `tp=1`，Miles 里 `tp=0`；
  每个非零 tp rank 计入的参数数 Lumen 388 vs Miles 260（差 128×3 = 384）。
- 梯度**本身是对的**（`wq_a` gsq 2.0816 vs Miles 2.0407），纯记账问题。
- 占 Lumen 总 sq 的 **22.5%**。
- 修法：改成 `set_tensor_model_parallel_attributes(self.weight, False, -1, 1)`。
- 对照 Miles：`LocalDuplicatedLinear` 直接 `nn.Parameter`，不打这个标记。

**缺陷 B：`LumenNorm` 没设 `weight.sequence_parallel`。**
SP 下每个 TP rank 只看到 1/tp_size 的序列，layernorm 的 wgrad 是偏和，要靠
Megatron `_allreduce_layernorm_grads` 按 `weight.sequence_parallel` 挑出来 allreduce。
`LumenNorm`（transformer 的 `input_layernorm` / `final_layernorm`）没设这个属性，
allreduce 被跳过。

- 证据：`final_layernorm.weight` gsq Lumen 0.419 vs Miles 6.243，比值 **0.067 ≈ 1/16**
  （梯度 1/4，平方 1/16）。
- `LocalRMSNorm` 有设（第 58 行），且 dsv4 的 `q_norm` / `kv_norm` 用 `config_no_sp`
  拿到 `False`——这是对的（它们看的是 gather 后的全序列），实测 `q_norm` 1.6466 vs 1.6141 已齐。
- 修法：`LumenNorm.__init__` 里补
  `self.weight.sequence_parallel = bool(getattr(config, "sequence_parallel", False))`。

两个缺陷方向相反、部分对冲（A 抬高、B 压低）。

**验修结果（2026-08-28 GPU 复验 SP RMSNorm）：**

| | lm loss | grad_norm |
|--|---------|-----------|
| Miles Triton bwd | 12.8385 | **17.03** |
| Lumen + dgrad + duplicated（LN 未修） | 12.8353 | **16.730** |
| Lumen + dgrad + duplicated + `LumenRMSNorm` SP | 12.8357 | **17.189** |

bucket 比值 Lumen/Miles 全部 ~1.00–1.02：`attn` 1.014、`shared_expert` 1.004、
`compressor` 1.010、**`norm` 1.007**（此前 0.476）。`final_layernorm` gsq
6.237 vs 6.243。全局 gn 比值 **1.009**。剩余 <1% 视为数值差（BF16 GEMM 等），
不再当作独立 bug。

### 1.6 Miles 代码修复（`dsv4_mi325` 已合）

`miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla.py`：

- **fwd 仍用 TileLang**。
- **bwd 改调 AIter Triton** `sparse_mla_bwd`（对 FP32 ref 正确）。
- TileLang LSE 是 log2，Triton 要自然对数 LSE：bwd 前用 Triton 再算一遍 `o/lse`。

单侧复验（`--tilelang-impl miles`）：small / flash 上 `dkv` 与 Triton **逐元素一致**（max diff 0）。

2-node 复验：`grad_norm` 31.09 → **17.13**（见 1.3）。

---

## 2. 其他机器上的环境

需要：

- 2 节点 × 8 GPU（MI308X/gfx942），或至少 1 GPU 做 kernel 单侧。
- Head/worker 都能 SSH，代码树一致（Lumen + Miles + AIter）。
- Docker 镜像：
  - Lumen：`zhangdanyangamd/lumen:dsv4-flash-308x-finetune`（或本集群等价 Lumen 镜像）
  - Miles：`rlsys/miles:rocm7.2-mi35x-dsv4`
- **AIter 必须带 DSV4 MHC / sparse MLA train API**。不要用缺 API 的 `/home/$USER/aiter`。  
  推荐：`AITER_DIR=$LUMEN_DIR/third_party/aiter`
- 本地 torch_dist ckpt（不要用 NFS 加载 532GB）。
- 对齐用 mock：Lumen `install_dsv4_safe_mock_data`；Miles `MILES_DSV4_LUMEN_MOCK=1`。

集群拓扑示例（原实验）：

```text
head   = 10.235.58.201
worker = $USER@<worker-host>
```

环境脚本（按本机改 IP/路径）：

```bash
source ~/dsv4-data/env_2node_d06-f01.sh   # 或自己的 2node env
export AITER_DIR="${LUMEN_DIR}/third_party/aiter"
export MASTER_ADDR=<head-ip>
export WORKER_SSH=$USER@<worker>
```

**PP 倒置（读 log / dump 时必看）：**

| | Lumen | Miles |
|--|-------|-------|
| PP0（浅层） | head | **worker** |
| PP3 last（深层 / loss / gn） | **worker** | **head** |

因此 Lumen 的 `iteration … lm loss … grad norm` 常在 **node1 worker log**；Miles 的 `train/grad_norm` 常在 **head native log**。

**Miles 1-step 陷阱：** 环境里若 `NUM_ROLLOUT=500`，会覆盖 `TRAIN_ITERS=1`。必须显式：

```bash
export TRAIN_ITERS=1 NUM_ROLLOUT=1
```

---

## 3. 复现：1-step 训练 gn / loss

先清容器：

```bash
docker rm -f lumen-dsv4-full-node0 lumen-dsv4-full-node1 \
  miles-dsv4-pretrain-native-node0 miles-dsv4-pretrain-native-node1 2>/dev/null || true
ssh "$WORKER_SSH" 'docker rm -f lumen-dsv4-full-node0 lumen-dsv4-full-node1 \
  miles-dsv4-pretrain-native-node0 miles-dsv4-pretrain-native-node1 2>/dev/null || true'
```

公共变量：

```bash
LOAD_CKPT=1 SKIP_PREPARE=1
GBS=8 SEQ_LEN=2048 TRAIN_ITERS=1 EVAL_ITERS=0 NUM_ROLLOUT=1
OPTIMIZER_OFFLOAD_FRACTION=0.6
PRETRAIN_LR=1e-6
AITER_DIR="${LUMEN_DIR}/third_party/aiter"
```

### 3.1 Lumen 默认 Triton（期望 gn ~16）

```bash
cd "$LUMEN_DIR"
LOAD_CKPT=1 SKIP_PREPARE=1 GBS=8 SEQ_LEN=2048 TRAIN_ITERS=1 EVAL_ITERS=0 \
  OPTIMIZER_OFFLOAD_FRACTION=0.6 PRETRAIN_LR=1e-6 \
  AITER_DIR="${LUMEN_DIR}/third_party/aiter" \
  bash examples/dsv4/launch_dsv4_flash_pretrain_2node.sh
```

Worker log 搜：

```text
lm loss: 1.28...E+01 | ... | grad norm: 16.3
```

### 3.2 Lumen 强制 TileLang（期望 gn ~35）

```bash
DSV4_FORCE_MLA_BACKEND=tilelang \
LOAD_CKPT=1 SKIP_PREPARE=1 GBS=8 SEQ_LEN=2048 TRAIN_ITERS=1 EVAL_ITERS=0 \
  OPTIMIZER_OFFLOAD_FRACTION=0.6 PRETRAIN_LR=1e-6 \
  AITER_DIR="${LUMEN_DIR}/third_party/aiter" \
  bash examples/dsv4/launch_dsv4_flash_pretrain_2node.sh
```

日志里应出现：

```text
[dsv4-mla] using TileLang sparse MLA (DSV4_FORCE_MLA_BACKEND=tilelang)
```

### 3.3 Miles（修 bwd 前 ~31；修后应再测）

```bash
cd "$MILES_DIR"
unset AITER_DIR   # 用镜像内 / 脚本默认；需要 MHC 时再指到 Lumen third_party/aiter
LOAD_CKPT=1 SKIP_PREPARE=1 GBS=8 SEQ_LEN=2048 TRAIN_ITERS=1 EVAL_ITERS=0 NUM_ROLLOUT=1 \
  OPTIMIZER_OFFLOAD_FRACTION=0.6 PRETRAIN_LR=1e-6 \
  IMAGE=rlsys/miles:rocm7.2-mi35x-dsv4 \
  bash examples/dsv4/launch_dsv4_flash_pretrain_native_2node.sh
```

Head log 搜：

```text
'train/lm loss': 12.83..., 'train/grad_norm': 31.0...
```

launch 脚本会 rsync `miles_plugins/` 到 worker。

---

## 4. 复现：激活 / 梯度 dump

### 4.1 打开 dump

Lumen：

```bash
DSV4_DUMP_TAG=lumen \
DSV4_DUMP_MHC_DX=1 \
DSV4_DUMP_MHC_DX_DIR=/root/models/dsv4_mhc_dx_lumen \
DSV4_DUMP_ATTN_IO=1 \
DSV4_DUMP_ATTN_IO_DIR=/root/models/dsv4_attn_io_lumen \
DSV4_DUMP_LAYER_ACT=0 DSV4_DUMP_GRAD_NORM=0 \
# 加上 3.1 或 3.2 的其余变量后 launch
```

Miles 同样：`DSV4_DUMP_TAG=miles`，目录改成 `..._miles`。

容器 `/root/models` → head `${MODEL_DIR}`，worker `${WORKER_MODEL_DIR}`。目录要可写（chmod 777 以免 root 留下不可写残留）。

**取 last-PP tp0：**

- Lumen last PP 在 **worker**：`lumen_w12_pp3_tp0.tsv`、`w12_pp3_tp0.txt`
- Miles last PP 在 **head**：`miles_w12_pp3_tp0.tsv`

layer id 在 dump 里是 `layer_id+1`（L43 = 最后一层）。attn-io 文本用 0-based，last-PP 常见 `layer=33..42`。

### 4.2 对比 Σg²（取每个 site **第一次**出现，避免多 step 污染）

```python
from pathlib import Path

def first_map(p):
    d = {}
    for line in Path(p).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 6 or parts[0] == "layer":
            continue
        k = (parts[0], parts[1], parts[2])
        if k not in d:
            d[k] = float(parts[5])
    return d

L = first_map("lumen_w12_pp3_tp0.tsv")
M = first_map("miles_w12_pp3_tp0.tsv")
for site in [
    "after_mla_o", "after_attn_out", "dkv_mla_in", "dkv_before_tp",
    "dkv_cmp_precat", "dkv_van_precat", "mla_dq", "after_wq_a",
    "after_wkv", "after_input_ln", "before_input_ln",
]:
    k = ("43", "attn", site)
    lv, mv = L.get(k), M.get(k)
    if lv and mv:
        print(f"{site:18s} L={lv:.6g} M={mv:.6g} M/L={mv/lv:.3f}")
```

关注点：

1. `after_mla_o` ≈ 1.00 → dO 齐。
2. `dkv_mla_in`：Triton vs TileLang 会先分叉。
3. `mla_dq` vs `after_wq_a`：区分 MLA dq 与 q 投影反传。

---

## 5. 复现：sparse MLA kernel 单侧（1 GPU，不必 2-node）

脚本：

- `Lumen/examples/dsv4/tools/compare_sparse_mla_kernels.py`
- `Lumen/examples/dsv4/tools/run_compare_sparse_mla_kernels.sh`

GPU 必须空闲（flash 的 Triton dkv workspace 约 2.5 GiB，卡满会 OOM）。

```bash
export AITER_DIR="${LUMEN_DIR}/third_party/aiter"
export IMAGE=zhangdanyangamd/lumen:dsv4-flash-308x-finetune
export GPU=0   # 选一张空闲卡

# Lumen 自带 TileLang bwd（期望 dkv cosine ~0.39 GAP）
bash "$LUMEN_DIR/examples/dsv4/tools/run_compare_sparse_mla_kernels.sh" --preset small
bash "$LUMEN_DIR/examples/dsv4/tools/run_compare_sparse_mla_kernels.sh" --preset flash

# Miles wrapper（dsv4_mi325 上 Triton bwd，期望 dkv 逐元素对齐）
bash "$LUMEN_DIR/examples/dsv4/tools/run_compare_sparse_mla_kernels.sh" \
  --preset small --tilelang-impl miles
bash "$LUMEN_DIR/examples/dsv4/tools/run_compare_sparse_mla_kernels.sh" \
  --preset flash --tilelang-impl miles
```

preset：

| 名 | shape | 是否 vs FP32 ref |
|----|--------|------------------|
| `small` | S=128, H=16, D=512, Skv=160, topk=128 | 是 |
| `flash` | S=2048, H=16, D=512, Skv=2560, topk=640 | 否（省显存） |
| `flash_win` | Skv=2064, topk=144 | 否 |

判据：`o/dq/dsink` cosine>0.99 且 sumsq 比在 0.7–1.4；`dkv` 同标准。TileLang 旧 bwd 会在 `dkv` 上 FAIL。

---

## 6. 建议排查顺序（新机器）

1. 1-step Lumen Triton vs Miles，确认 loss~12.84、gn 16 vs 31（若 Miles 已是 Triton bwd，gn 应变近 Lumen）。
2. 打开 `DSV4_DUMP_MHC_DX` + `DSV4_DUMP_ATTN_IO`，比 last-PP `after_mla_o` 与 `dkv_mla_in`。
3. Lumen `DSV4_FORCE_MLA_BACKEND=tilelang`：gn 应升到 ~35，dkv dump 贴 Miles（旧 TileLang）。
4. 跑 kernel compare `--preset small`：确认 TileLang `dkv` vs ref 失败、Triton 通过。
5. `--tilelang-impl miles`：确认 Miles 补丁后 `dkv` 与 Triton 一致。
6. 仍有 gn 差再追 `after_wq_a`（q_norm / wq_b / RoPE），不要回头换 MHC。

不要做：把 MHC 换回 TileKernels；不要用缺 MHC API 的 AIter；不要只靠 `V4_SPARSE_MLA_BACKEND` 以为切了 Lumen MLA。

---

## 7. 关键代码位置

| 作用 | 路径 |
|------|------|
| Lumen MLA 选择 | `lumen/ops/dsv4/sparse_mla.py`（`DSV4_FORCE_MLA_BACKEND`） |
| Lumen Triton 包装 | `lumen/kernels/dsv4/sparse_mla/triton_sparse_mla.py` |
| Lumen/Miles TileLang autograd | `.../tilelang_sparse_mla.py` |
| 正确 dkv bwd | `aiter/ops/triton/attention/sparse_mla_dsv4_train.py` → `sparse_mla_bwd` |
| Lumen dump 位点 | `lumen/models/dsv4/megatron/deepseek_v4.py`、`ops/hyper_connection.py` |
| Miles dump 位点 | `miles_plugins/models/deepseek_v4/deepseek_v4.py` |

---

## 8. 原实验数字备忘

Lumen worker 1-step（Triton）：`lm loss 1.283689E+01`，`grad norm: 16.327`。  
Lumen TileLang：`lm loss 1.283761E+01`，`grad norm: 34.981`。  
Miles step 0（旧 TileLang bwd）：`lm loss 12.838`，`grad_norm 31.089`。
