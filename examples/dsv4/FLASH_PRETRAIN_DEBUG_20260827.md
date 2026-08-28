# DSV4 Flash 对齐：2026-08-27 排查纪要

今天在 2×8 MI308X（TP4 / PP4 / EP4，Flash 43L，GBS=8，SEQ=2048，1 step）上，把 Lumen 和 Miles 的 **反向 `grad_norm` 从 ~2× 差拆到具体 kernel / 模块**。前向本来就齐。复现命令见同目录 [FLASH_PRETRAIN_ALIGN.md](./FLASH_PRETRAIN_ALIGN.md)。

分支：两边都是 `dsv4_mi325`。

---

## 一句话

- **Miles `grad_norm` ~31 vs Lumen ~16**：几乎全是 TileLang sparse MLA backward 的 **`dkv` 算错**（fwd 没问题）。
- 修完 Miles bwd 之后，两边还差一截，差在 **Lumen 自己**：`wq_b` 缺 TP dgrad allreduce、duplicated 权重被当成 TP 重复计入 `grad_norm`、SP RMSNorm 的 wgrad 没 allreduce。
- 2026-08-28 补上 `LumenRMSNorm` 的 SP 标记并复验：**Lumen 17.189 vs Miles 17.03**（比值 **1.009**）。`norm` bucket 0.48 → **1.007**。反向 `grad_norm` 按参数桶已对齐。

---

## 数字怎么走过来的

同一 ckpt、同一 mock、lr=1e-6、optimizer offload 0.6。`lm loss` 全程 ≈ **12.84**。

| 阶段 | Lumen gn | Miles gn | 说明 |
|------|----------|----------|------|
| 开始（默认 kernel） | ~16 | **~31** | Lumen Triton MLA；Miles TileLang MLA |
| Lumen 强制 TileLang | ~35 | ~31 | 两边 `dkv` 对齐，说明差在 kernel 不是 CE/LN/MHC |
| Miles 改 Triton bwd | ~16.40 | **17.13** | TileLang `dkv` 是 2× 主因 |
| Lumen 补 dgrad allreduce | **19.09** | 17.03 | `after_wq_a` 从 0.24× 回到 ~1.00；gn 暂时变高 |
| Lumen 再修 duplicated 记账 | **16.730** | 17.03 | `attn` bucket 1.30 → 1.007；LN 仍偏小 |
| Lumen 再修 `LumenRMSNorm` SP | **17.189** | 17.03 | `norm` 0.48 → 1.007；全局 1.009× |

PP 是反的：Lumen last-PP / `grad norm` 在 **worker log**；Miles 在 **head log**。

---

## 发现 1：前向没问题

- last-PP `after_mla_o` / `after_attn_out` / q·kv·o·topk 约 **1.00×**。
- 要对齐必须同时满足：同一 ckpt、Miles `MILES_DSV4_LUMEN_MOCK=1`、fused vocab-parallel CE、freeze gate / hash skip。
- `V4_SPARSE_MLA_BACKEND` **不会**改 Lumen kernel；Lumen 只认 `DSV4_FORCE_MLA_BACKEND`。

---

## 发现 2：Miles TileLang sparse MLA 的 `dkv` 是错的

单侧脚本：`examples/dsv4/tools/compare_sparse_mla_kernels.py`（相对 FP32 ref）。

| 张量 | TileLang vs ref | Triton vs ref |
|------|-----------------|---------------|
| `o` / `dq` / `dsink` | cosine ≈ 1 | cosine ≈ 1 |
| **`dkv`** | **cosine ≈ 0.39**（能量却 ≈ 1） | cosine ≈ 1 |

Flash shape（seq=2048，heads=16，dim=512，kv=2560，topk=640）同样结论。`dkv` 是 MQA，shape 与 `kv` 相同：last-PP 入口 `(1, 2560, 512)`。

训练侧对照：

- 两边都 Triton：`dkv_mla_in` ≈ 1.00。
- 两边都 TileLang：`dkv_*` / `after_wkv` ≈ 1.00，但相对 ref 两边一起错。
- 只 Miles TileLang：`dkv_mla_in` Miles/Lumen ≈ 1.63×，残差往后放大，全局 gn ~31。

**改法（已合 Miles `cd6bf14f4`）：** `tilelang_sparse_mla.py` 保持 TileLang **fwd**，bwd 调 AIter `sparse_mla_bwd`。TileLang LSE 是 log2，Triton 要自然对数，所以 bwd 前用 Triton 再算一遍 `o/lse`。修完 2-node gn **31.09 → 17.13**。

---

## 发现 3：Lumen `wq_b` 缺 TP dgrad allreduce（真梯度错误）

`LumenColumnParallelLinear` 抄了 Megatron 的分支：`allreduce_dgrad=True` 时不走 `copy_to_tensor_model_parallel_region`，指望融合 GEMM 的 autograd 去做 allreduce。Lumen 的 `_do_gemm` **没有这个参数**，allreduce 谁都没做。

只在 `tp_size > 1` 且 `sequence_parallel=False` 时触发。正常 transformer 全程开 SP，所以只有 DSV4 attention 里用 `config_no_sp` 建的 **`wq_b` / `wo_a`** 中枪。

现象：L43 `after_wq_a` / `dx_wq` Lumen/Miles ≈ **0.24（≈ 1/TP）**，而 `mla_dq` 已经 ~1.00。以前误以为缺口在 `q_norm → RoPE`，其实是 column-parallel 的输入梯度没在 TP 间加起来。

Miles 对照：gfx942 上 `te_or_local.py` 禁 TE，走 Megatron 原生 `ColumnParallelLinear`，这条路是对的。

修法：即使 `allreduce_dgrad` 也为 True，仍走 `copy_to_tensor_model_parallel_region`（fwd identity / bwd allreduce）。

修完 L43：`after_wq_a` **0.237 → 1.001**，`dx_wq` **0.265 → 0.999**，`after_mla_o` 不变。全 43 层激活比值从 0.09–1.14 收到约 **0.83–1.08**。L22（PP1 最后一层）全站点 ≈ 0.64，像 PP 边界，没深挖。

文件：`lumen/modules/parallel_linear.py`。2-node launcher 必须 rsync 这个文件，否则 worker 拿不到补丁。

---

## 发现 4：`grad_norm` 对不上，不一定是激活梯度不对

dgrad 修好后激活已经齐，但 Lumen gn **19.09** vs Miles **17.03**。逐层激活 Lumen 反而略**低**。用 `DSV4_DUMP_GRAD_NORM=1` 的 per-param `gsq` 拆：`sqrt(sum(gsq[inc]))` 能精确复现 Megatron 的 `grad norm`。

当时 Lumen 分桶（相对 Miles）：

| bucket | gn 比值 | 含义 |
|--------|---------|------|
| attn | **1.295** | 重复计数 |
| shared_expert / compressor / expert | ~1.00 | 权重梯度已齐 |
| **norm** | **0.479** | LN wgrad 偏小 |

比较脚本：`examples/dsv4/tools/compare_grad_norm_dump.py`。

---

## 发现 5：duplicated 线性层被标成 tensor-parallel（记账错误）

`LumenDuplicatedLinear` 调了 `set_tensor_model_parallel_attributes(weight, True, …)`。`wq_a` / `wkv` 每个 TP rank 上一份完整副本，标成 TP 后 Megatron 的 gn **四个 rank 各数一遍**。

证据：同一份权重 Lumen dump `tp=1`、Miles `tp=0`；非 tp0 计入参数数 388 vs 260。`wq_a` 的 gsq 本身 2.08 vs 2.04，**梯度对、范数虚高**。约占当时 Lumen 总 sq 的 22.5%。

Miles `LocalDuplicatedLinear` 就是普通 `nn.Parameter`，没有这个标记。

修法：`set_tensor_model_parallel_attributes(self.weight, False, -1, 1)`。修完 gn **19.09 → 16.730**，`attn` bucket **1.007**。

文件：`lumen/models/dsv4/megatron/layers.py`。

---

## 发现 6：SP layernorm wgrad 没 allreduce，而且打错了 class

SP 下每个 TP rank 只看到 1/4 序列，LN 的 wgrad 是偏和，要靠 Megatron `_allreduce_layernorm_grads` 认 `weight.sequence_parallel`。

`final_layernorm.weight` gsq：Lumen 0.42 vs Miles 6.24，比值 **≈ 0.067 ≈ 1/16**（梯度 1/4，再平方）。`q_norm` / `kv_norm` 用 `config_no_sp`，看的是 gather 后的全序列，它们是齐的（1.65 vs 1.61）。

第一刀打在 `layers.LumenNorm` 上，**没生效**。默认 `LUMEN_DSV4_LOCAL_RMSNORM=0` 时，transformer 的 `input_layernorm` / `final_layernorm` 走的是：

`spec_provider._LumenNorm` → **`lumen.ops.normalization.LumenRMSNorm`**

`LumenRMSNorm` 原来不收 `config`，也不设 `sequence_parallel`。已改 factory 传入 config，并在 `LumenRMSNorm` / `LumenLayerNorm` 上打标记。**2026-08-28 复验**：`norm` bucket 1.007，`final_layernorm` gsq 6.237 vs 6.243，全局 gn **17.189 vs 17.03**。

---

## 还没做完

1. ~~GPU 复验 `LumenRMSNorm.sequence_parallel`~~ **已做（2026-08-28）**：gn 17.189 vs 17.03，`norm` 1.007。
2. L22 PP 边界激活 dump ≈ 0.64 未解释（**不影响**全局 `grad_norm` 对齐）。
3. TileLang `dkv` kernel 本身没修，只是 Miles bwd 绕开了。

---

## 当天改过的文件（已 commit）

**Lumen `251afe0`**

- `lumen/modules/parallel_linear.py` — dgrad allreduce
- `lumen/models/dsv4/megatron/layers.py` — duplicated 非 TP；`LumenNorm` SP 标记（对 transformer LN 无效，保留无害）
- `lumen/ops/normalization/rmsnorm.py` / `layernorm.py` + `lumen/models/spec_provider.py` — 真正生效的 LN 标记（未 GPU 复验）
- `examples/dsv4/launch_dsv4_flash_pretrain_2node.sh` — rsync 上述文件
- `examples/dsv4/tools/compare_grad_norm_dump.py`、本目录对齐文档

**Miles `cd6bf14f4` + `b00377770`**

- `miles_plugins/.../tilelang_sparse_mla.py` — Triton bwd
- `examples/dsv4/FLASH_PRETRAIN_ALIGN.md`
