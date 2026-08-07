# DSV4 Flash 全模型 Runbook（2-node MI308X）

操作手册：在 **head + worker** 双节点 16 GPU 上跑 Lumen native GRPO full finetune（43 层）。详细脚本说明见 [README.md](./README.md)。

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

**Worker 上还需**：`${LUMEN_DIR}`、`${TILEKERNELS_DIR}` 与 head 同步。

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
V4_INDEXER_IMPL=aiter V4_SPARSE_MLA_BACKEND=triton MHC_BACKEND=triton \
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
| `MHC_BACKEND` | `triton` | 需挂载 TileKernels |
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
V4_INDEXER_IMPL=aiter V4_SPARSE_MLA_BACKEND=triton MHC_BACKEND=triton \
OPTIMIZER_OFFLOAD_FRACTION=0.75 \
NCCL_IB_GDR_LEVEL=0 NCCL_NET_GDR_LEVEL=LOC MEGATRON_NO_BATCH_P2P_COMM=1 \
HSA_OVERRIDE_GFX_VERSION=9.4.2 NCCL_SOCKET_IFNAME=ens14np0 GLOO_SOCKET_IFNAME=ens14np0 \
IMAGE=lumen/dsv4-lumen:mi308x \
DSV4_PROFILE=flash bash examples/dsv4/run_dsv4.sh
```

Worker：

```bash
cd "${LUMEN_DIR}"
NODE_RANK=1 MASTER_ADDR=<head-ip> \
MODEL_DIR=/mnt/nvme0n1/${USER}/models \
# 其余 env 与 head 相同（GBS/SEQ/NCCL/…）
DSV4_PROFILE=flash bash examples/dsv4/run_dsv4.sh
```

---

## 5. 日志与监控

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

### 5.1 训练成功标志

- `rollout/num_samples`, `train/loss`, `perf/actor_train_time`
- 结束：`=== [done] Lumen DSV4 flash native GRPO finetune completed ===`

### 5.2 正常但耗时的阶段

- Checkpoint 加载：`q_norm/kv_norm ... will skip`（需 `LUMEN_DSV4_SKIP_OPTIONAL_NORMS=1`，默认开启）
- Optimizer CPU offload 初始化：host 内存高，可能数十分钟无新行

---

## 6. 故障排查

| 现象 | 原因 / 处理 |
|------|------------|
| Checkpoint 加载极慢 | 确认 `MODEL_DIR` / `WORKER_MODEL_DIR` 指向本地 NVMe，而非 `${DATA_ROOT}/models`（NFS） |
| `TRAIN_ITERS: unbound variable` | 使用最新 `preflight_dsv4_flash_multinode.sh` |
| rollout 数据缺失 | 确认双节点 `FAKE_ROLLOUT_DATA=${DATA_ROOT}/models/fake_rollout.pt`；或 `SMOKE_LEGACY_FAKE_ROLLOUT=1` |
| worker rank OOM | worker GPU 被其他进程占用；`rocm-smi --showpids` 确认并释放 |
| NCCL hang / 配置不一致 | 同一 `launch_dsv4_2node.sh` 启动；preflight 校验 GBS/TP/NCCL |
| `NET/IB : Unable to open device mlx5_*` | IB 不可用时的 WARN，可能 fallback 到 socket |
| Worker unreachable | 检查 SSH key、`WORKER_SSH`、reservation / 安全组 |

---

## 7. 相关脚本

```text
examples/dsv4/
├── runbook.md                              # 本文档
├── launch_dsv4_2node.sh                    # 双节点 finetune 一键启动（推荐）
├── run_dsv4.sh                             # 单 rank launcher
├── run_dsv4_inner.sh                       # 容器内 torchrun GRPO
├── finetune_dsv4_megatron.py               # Python 入口
├── dsv4_finetune_common.sh                 # batch / ckpt / rollout helpers
├── tools/gen_fake_rollout_data.py          # fake_rollout.pt
├── preflight_dsv4_flash_multinode.sh       # 双节点配置校验
└── dsv4_paths.sh                           # MODEL_DIR / LOG_DIR 等
```
