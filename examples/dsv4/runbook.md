# DSV4 Flash 全模型 Runbook（2-node MI300X / MI308X）

操作手册：在 **head + worker** 双节点 16 GPU 上跑 Lumen native **GRPO finetune** 与 **Megatron pretrain**（43 层）。详细脚本说明见 [README.md](./README.md)。

---

## 1. 集群与环境

| 项目 | 说明 |
|------|------|
| Head | `NODE_RANK=0`，IP = `${MASTER_ADDR}` |
| Worker | `NODE_RANK=1`，SSH = `${WORKER_SSH}` |
| Docker 镜像 | `lumen/dsv4-lumen:mi308x` |
| 并行 | TP=4, PP=4, EP=4（11+11+11+10 层） |
| 网络 | 按集群设置 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`（MI308X Banff 常用 `ens14np0`） |
| NCCL workaround | `NCCL_IB_GDR_LEVEL=0`, `NCCL_NET_GDR_LEVEL=LOC`, `MEGATRON_NO_BATCH_P2P_COMM=1` |

**Worker 上还需**：`${LUMEN_DIR}` 与 `/workspace/aiter` 中的 AIter 部署与 head 同步。

### 1.1 本地 checkpoint（推荐，默认 launch 示例）

Head/worker 使用各自 NVMe 上的 ckpt（**不要**用 NFS 加载 532GB dist ckpt，否则 init 极慢）：

| 节点 | launch 变量 | 示例 |
|------|-------------|------|
| head | `MODEL_DIR` | `/data1/${USER}/models` |
| worker | `WORKER_MODEL_DIR` | `/mnt/nvme0n1/${USER}/models` |

容器内均映射为 `/root/models/${MODEL_NAME}_torch_dist`。

---

## 2. 路径（NFS + 本地）

默认由 `dsv4_paths.sh` 解析（`DATA_ROOT` 自动探测 `/nfs/data/${USER}` 等）：

| 用途 | 宿主机路径 | 容器内路径 |
|------|-----------|-----------|
| 模型目录（**默认推荐本地 NVMe**） | head: `/data1/${USER}/models`<br>worker: `/mnt/nvme0n1/${USER}/models` | `/root/models` |
| 数据集 | `${DATA_DIR}` | `/root/datasets` |
| 日志 | `${LOG_DIR}` | — |
| Lumen 代码 | `${LUMEN_DIR}` | `/workspace/Lumen` |
| **共享 rollout（双节点，NFS）** | `${DATA_ROOT}/models/fake_rollout.pt` | 同路径（`DATA_ROOT` bind-mount） |

`${MODEL_DIR}` 未显式设置时 `dsv4_paths.sh` 会落到 `${DATA_ROOT}/models`（NFS）——仅适合小模型或已有 ckpt；**43L flash finetune 请始终指定本地 `MODEL_DIR` / `WORKER_MODEL_DIR`**。

### 2.1 预训练 checkpoint（finetune **加载**）

| 路径 | 说明 |
|------|------|
| `${MODEL_DIR}/${MODEL_NAME}_torch_dist` | 主 checkpoint（torch_dist） |
| `${MODEL_DIR}/${MODEL_NAME}_torch_dist_hc${DSV4_HC_MULT}` | fallback |

校验：`latest_checkpointed_iteration.txt` 存在即可。

### 2.2 GRPO rollout 数据

| 路径 | 说明 |
|------|------|
| `${DATA_ROOT}/models/fake_rollout.pt` | debug-train-only 假 rollout（双节点共享） |

双节点时 rank0 在 **NFS `${DATA_ROOT}/models/`** 生成/复用；worker 即使 ckpt 在本地 NVMe，也读同一份 NFS rollout。已存在则跳过。

单节点默认：`/root/models/fake_rollout.pt`。

### 2.3 Finetune **输出** checkpoint

**默认不保存**。脚本未传 `--save`，`--save-interval 1000000` 等价于关闭。

如需保存 finetune 权重，在 `dsv4_finetune_common.sh` 的 `DSV4_FINETUNE_TORCHRUN_ARGS` 中增加例如：

```bash
--save /root/models/DeepSeek-V4-Flash-FP8-finetune_torch_dist
--save-interval 1
```

---

## 3. 启动前检查

### 3.1 同步代码到 worker

在 **head** 执行：

```bash
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'third_party/aiter/**/build' --exclude '.nfs*' \
  "${LUMEN_DIR}/" "${WORKER_SSH}:${LUMEN_DIR}/" \
  -e "ssh ${SSH_KEY:+-i ${SSH_KEY} -o IdentitiesOnly=yes} -o BatchMode=yes"
```

### 3.2 确认 worker 8 卡空闲

```bash
ssh ${SSH_KEY:+-i ${SSH_KEY} -o IdentitiesOnly=yes} -o BatchMode=yes "${WORKER_SSH}" \
  'for i in 0 1 2 3 4 5 6 7; do echo -n "GPU$i: "; \
   rocm-smi -d $i --showmeminfo vram 2>/dev/null | grep Used; done'
```

### 3.3 清理旧容器

```bash
docker rm -f lumen-dsv4-flash-finetune-node0 lumen-dsv4-flash-finetune-node1 2>/dev/null || true
ssh ${SSH_KEY:+-i ${SSH_KEY} -o IdentitiesOnly=yes} -o BatchMode=yes "${WORKER_SSH}" \
  'docker rm -f lumen-dsv4-flash-finetune-node0 lumen-dsv4-flash-finetune-node1 2>/dev/null || true'
```

---

## 4. 启动：DSV4 43 层 GRPO full finetune

在 **head** 执行（推荐一键双节点）：

```bash
cd "${LUMEN_DIR}"

MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-ip> \
MODEL_DIR=/data1/${USER}/models \
WORKER_MODEL_DIR=/mnt/nvme0n1/${USER}/models \
DATA_ROOT=/nfs/data/${USER} \
SKIP_PREPARE=1 DSV4_HC_MULT=4 \
V4_INDEXER_IMPL=aiter V4_SPARSE_MLA_BACKEND=triton \
OPTIMIZER_OFFLOAD_FRACTION=0.75 \
NCCL_IB_GDR_LEVEL=0 NCCL_NET_GDR_LEVEL=LOC MEGATRON_NO_BATCH_P2P_COMM=1 \
HSA_OVERRIDE_GFX_VERSION=9.4.2 NCCL_SOCKET_IFNAME=ens14np0 GLOO_SOCKET_IFNAME=ens14np0 \
IMAGE=lumen/dsv4-lumen:mi308x \
bash examples/dsv4/launch_dsv4_2node.sh
```

默认 finetune batch：**`GBS=256`**、**`SEQ_LEN=4096`**、**`NUM_ROLLOUT=10`**（`launch_dsv4_2node.sh` / `dsv4_finetune_common.sh`）。

**可选 bisect smoke**（连通性/PP 调试，非默认生产配置）：

```bash
GBS=8 DSV4_KEEP_GBS=1 SEQ_LEN=512 DSV4_KEEP_SEQ_LEN=1 NUM_ROLLOUT=2 \
ROLLOUT_N_PROMPTS=1 ROLLOUT_N_PER_PROMPT=8 SMOKE_LEGACY_FAKE_ROLLOUT=1 \
# …其余 env 同上（含 MODEL_DIR / WORKER_MODEL_DIR）…
bash examples/dsv4/launch_dsv4_2node.sh
```

### 4.1 关键参数

| 变量 | 默认/推荐 | 说明 |
|------|----------|------|
| `GBS` | **`256`** | 必须等于 rollout 样本数（32×8） |
| `SEQ_LEN` | **`4096`** | Megatron 训练序列长度 |
| `MBS` | `1` | micro-batch |
| `NUM_ROLLOUT` | `10` | GRPO rollout / train iters |
| `DSV4_HC_MULT` | `4` | MHC 乘数 |
| `SKIP_PREPARE` | `1`（launch） | 跳过 HF→torch_dist 转换 |
| `MODEL_DIR` | **`/data1/${USER}/models`（head）** | 本地 NVMe ckpt；勿用 NFS 加载全模型 |
| `WORKER_MODEL_DIR` | **`/mnt/nvme0n1/${USER}/models`** | worker 本地 ckpt |
| `DATA_ROOT` | `/nfs/data/${USER}` | 共享 rollout 路径 |
| `V4_INDEXER_IMPL` | `aiter` | DSA indexer（aiter triton kernel） |
| `V4_SPARSE_MLA_BACKEND` | `triton` | sparse MLA |
| MHC | AIter | 直接使用 AIter DSV4 fused API |
| `MEGATRON_NO_BATCH_P2P_COMM` | `1` | 避免 PP P2P hang |
| `NCCL_IB_GDR_LEVEL` | `0` | IB GDR workaround |
| `NCCL_NET_GDR_LEVEL` | `LOC` | 配合上项 |
| `OPTIMIZER_OFFLOAD_FRACTION` | `0.75` | CPU Adam offload |
| `SSH_KEY` | — | worker SSH 私钥路径（可选，不设则用 ssh-agent 默认 key） |

### 4.2 单节点手动启动（备用）

Head：

```bash
cd "${LUMEN_DIR}"
NODE_RANK=0 MASTER_ADDR=<head-ip> \
MODEL_DIR=/data1/${USER}/models \
SKIP_PREPARE=1 DSV4_HC_MULT=4 \
V4_INDEXER_IMPL=aiter V4_SPARSE_MLA_BACKEND=triton \
OPTIMIZER_OFFLOAD_FRACTION=0.75 \
NCCL_IB_GDR_LEVEL=0 NCCL_NET_GDR_LEVEL=LOC MEGATRON_NO_BATCH_P2P_COMM=1 \
HSA_OVERRIDE_GFX_VERSION=9.4.2 NCCL_SOCKET_IFNAME=ens14np0 GLOO_SOCKET_IFNAME=ens14np0 \
IMAGE=lumen/dsv4-lumen:mi308x \
DSV4_PROFILE=flash bash examples/dsv4/dsv4_launch.sh
```

Worker：

```bash
cd "${LUMEN_DIR}"
NODE_RANK=1 MASTER_ADDR=<head-ip> \
MODEL_DIR=/mnt/nvme0n1/${USER}/models \
# 其余 env 与 head 相同（GBS/SEQ/NCCL/…）
DSV4_PROFILE=flash bash examples/dsv4/dsv4_launch.sh
```

---

## 5. 43L Pretrain（native Megatron，mock data）

Pretrain 使用 **`--mock-data`**，不需要 `${DATA_ROOT}/datasets/` 下真实语料。`pretrain_dsv4_megatron.py` 入口会调用 `install_dsv4_safe_mock_data()`，避免默认 mock token 触发 hash MoE `tid2eid=-1`。

### 5.1 标准配置

| 项 | 值 |
|----|-----|
| 入口 | `launch_dsv4_2node.sh`（`DSV4_MODE=pretrain`）→ `dsv4_launch.sh` → `run_dsv4_pretrain_inner.sh` |
| Checkpoint | `${MODEL_NAME}_torch_dist`（本地 NVMe，head/worker 各自 `MODEL_DIR` / `WORKER_MODEL_DIR`） |
| Batch | GBS=8, MBS=1, **seq_len=2048** |
| LR | 1e-6 constant（`run_dsv4_pretrain_inner.sh` 默认） |
| Optimizer | Adam + CPU offload fraction=0.75 |
| Kernels | MLA=triton, MHC=AIter, indexer=aiter, GEMM BF16 tuned CSV |
| Expert bias | **默认关闭**（`DSV4_ENABLE_EXPERT_BIAS=1` 才开启；converted ckpt 常缺 `expert_bias` shard） |
| Rerun | flash 默认 `RERUN_MODE=disabled`（避免 Megatron validate_results 重复跑 iter） |

### 5.2 一键启动（双节点）

```bash
cd "${LUMEN_DIR}"

LOAD_CKPT=1 SKIP_PREPARE=1 TRAIN_ITERS=10 \
MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-ip> \
MODEL_DIR=/data1/${USER}/models \
WORKER_MODEL_DIR=/mnt/nvme0n1/${USER}/models \
DATA_ROOT=/nfs/data/${USER} LOG_DIR=/nfs/data/${USER}/logs \
AITER_DIR=${LUMEN_DIR}/third_party/aiter \
V4_INDEXER_IMPL=aiter V4_SPARSE_MLA_BACKEND=triton \
OPTIMIZER_OFFLOAD_FRACTION=0.75 \
NCCL_IB_GDR_LEVEL=0 NCCL_NET_GDR_LEVEL=LOC MEGATRON_NO_BATCH_P2P_COMM=1 \
HSA_OVERRIDE_GFX_VERSION=9.4.2 NCCL_SOCKET_IFNAME=ens14np0 GLOO_SOCKET_IFNAME=ens14np0 \
IMAGE=lumen/dsv4-lumen:mi308x \
DSV4_MODE=pretrain bash examples/dsv4/launch_dsv4_2node.sh
```

4-layer smoke（单节点）：

```bash
DSV4_MODE=pretrain DSV4_PROFILE=4layer LOAD_CKPT=1 SKIP_PREPARE=1 TRAIN_ITERS=20 \
bash examples/dsv4/dsv4_launch.sh
```

### 5.3 正确性验证 ladder

| Tier | 做法 | PASS 标准 |
|------|------|-----------|
| **0** | §3 同步代码、GPU 空闲、本地 NVMe ckpt | head/worker manifest 一致 |
| **1** | 4L pretrain，20 step | `[done]`，0 NaN |
| **2** | 43L 双节点，`TRAIN_ITERS=2`，`LOAD_CKPT=1` | iter1/2 **lm loss ~17.x**，**0 NaN**，grad norm 有界 |
| **3** | 43L 双节点，`TRAIN_ITERS=100+`，LR=1e-6 | loss 缓降、0 NaN（**长程稳定性**另计） |

**Loss 量级（load ckpt + mock data，GBS=8）：**

| 场景 | 预期 lm loss |
|------|-------------|
| Random init + mock | ~log(vocab) ≈ 11.8 |
| Load ckpt + mock | **~17–18**（mock token 分布 ≠ 预训练分布，CE 偏高但正常） |
| LR=1e-6，20+ step | 缓降（如 17.6→17.0） |

不与 Megatron-Bridge 或其他框架的绝对 loss 直接对比；看 **0 NaN + 趋势 + 短程可复现**。

**检查命令：**

```bash
grep 'lm loss' "${LOG_DIR}"/lumen_dsv4_flash_pretrain_node1_*.log | tail -5
grep 'nan iterations' "${LOG_DIR}"/lumen_dsv4_flash_pretrain_node1_*.log | tail -5
grep '\[done\]' "${LOG_DIR}"/lumen_dsv4_flash_pretrain_node*_*.log
```

**历史验证结论（2026-08）：** Tier 2 在 Banff 双节点 PASS（iter1/2 loss 17.608→17.654，Δ≈0.05）；Tier 3 有 loss 的 step 全部 0 NaN。长程 crash（Bus error / DRAM ECC）为 **硬件/稳定性** 问题，不是 loss 算错。

### 5.4 Pretrain 日志

| 日志 | 路径 |
|------|------|
| Head 训练 | `${LOG_DIR}/lumen_dsv4_flash_pretrain_node0_*.log` |
| Worker 训练 | `${LOG_DIR}/lumen_dsv4_flash_pretrain_node1_*.log` |
| Launch | `${LOG_DIR}/lumen_dsv4_flash_launch_{head,worker}_*.log` |

---

## 6. 日志与监控（Finetune）

| 日志 | 路径 |
|------|------|
| Head 训练 | `${LOG_DIR}/lumen_dsv4_flash_finetune_node0_*.log` |
| Worker 训练 | `${LOG_DIR}/lumen_dsv4_flash_finetune_node1_*.log` |
| Launch head | `${LOG_DIR}/lumen_dsv4_flash_finetune_launch_head_*.log` |
| Launch worker | `${LOG_DIR}/lumen_dsv4_flash_finetune_launch_worker_*.log` |
| Preflight | `${LOG_DIR}/.dsv4_preflight/runs/<PREFLIGHT_ID>/` |

```bash
tail -f "${LOG_DIR}"/lumen_dsv4_flash_finetune_node0_*.log
docker ps --filter name=lumen-dsv4-flash-finetune
```

### 6.1 训练成功标志

- `rollout/num_samples`, `train/loss`, `perf/actor_train_time`
- 结束：`=== [done] Lumen DSV4 flash native GRPO finetune completed ===`

### 6.2 正常但耗时的阶段

- Checkpoint 加载：`q_norm/kv_norm ... will skip`（需 `LUMEN_DSV4_SKIP_OPTIONAL_NORMS=1`，默认开启）
- Optimizer CPU offload 初始化：host 内存高，可能数十分钟无新行

---

## 7. 故障排查

| 现象 | 原因 / 处理 |
|------|------------|
| Checkpoint 加载极慢 | 确认 `MODEL_DIR` / `WORKER_MODEL_DIR` 指向本地 NVMe，而非 `${DATA_ROOT}/models`（NFS） |
| `expert_bias` load 失败 | converted ckpt 缺 shard；保持 `DSV4_ENABLE_EXPERT_BIAS=0`（默认） |
| `TRAIN_ITERS: unbound variable` | 使用最新 `preflight_dsv4_flash_multinode.sh` |
| rollout 数据缺失 | 确认双节点 `FAKE_ROLLOUT_DATA=${DATA_ROOT}/models/fake_rollout.pt`；或 `SMOKE_LEGACY_FAKE_ROLLOUT=1` |
| worker rank OOM | worker GPU 被其他进程占用；`rocm-smi --showpids` 确认并释放 |
| NCCL hang / 配置不一致 | 同一 launch 脚本启动；preflight 校验 GBS/TP/NCCL；head/worker `examples/dsv4/` rsync 一致 |
| `NET/IB : Unable to open device mlx5_*` | IB 不可用时的 WARN，可能 fallback 到 socket |
| Worker unreachable | 检查 SSH key、`WORKER_SSH`、reservation / 安全组 |
| Pretrain `not found tuned config` | GEMM CSV 按 **runtime `cu_num`** 查表（MI300X=304，MI308X=80）；在对应节点 `CU_NUM=<n> bash run_tune_dsv4_bf16_gemm_flash2node.sh` 补 tune |
| Pretrain Bus error / DRAM ECC | 多在 iter 2+ optimizer step；优先 bisect：`OPTIMIZER_OFFLOAD_FRACTION=0`、`LUMEN_DSV4_GEMM_BF16=0`；查 `dmesg`/ECC |
| Mock data GPU fault | 确认 `install_dsv4_safe_mock_data()` 已执行（`pretrain_dsv4_megatron.py` 默认） |

---

## 8. 相关脚本

```text
examples/dsv4/
├── runbook.md                              # 本文档
├── dsv4_launch.sh                          # 统一 Docker launcher（唯一入口）
├── launch_dsv4_2node.sh                    # 双节点一键启动（DSV4_MODE=finetune|pretrain）
├── run_dsv4_pretrain_inner.sh              # 容器内 pretrain/profile torchrun
├── run_dsv4_inner.sh                       # 容器内 GRPO finetune torchrun
├── dsv4_megatron_args.sh                   # 4layer/flash 模型参数
├── megatron_patch/                         # ROCm Megatron 模块化 patch
├── finetune_dsv4_megatron.py               # GRPO finetune 入口
├── pretrain_dsv4_megatron.py               # Pretrain 入口（含 profiler hook）
├── dsv4_finetune_common.sh                 # finetune batch / ckpt / rollout
├── dsv4_pretrain_common.sh                 # pretrain ckpt / load helpers
├── preflight_dsv4_flash_multinode.sh       # 双节点配置校验
└── dsv4_paths.sh                           # MODEL_DIR / LOG_DIR 等

lumen/models/dsv4/                         # compressor, hyper_connection, megatron/
lumen/tools/dsv4/                          # ckpt / rollout / convert 工具
lumen/ops/dsv4/                            # 无状态 op
lumen/kernels/dsv4/                        # GPU kernel
```
