# DSV4 on MI300X / MI308X — GRPO finetune & Megatron pretrain

**入口**：`dsv4_launch.sh` — 统一 Docker launcher（`DSV4_MODE=finetune|pretrain|profile`）。

模型实现：`lumen/models/dsv4/`

双节点一键启动见 [runbook.md](./runbook.md)。

---

## 训练路径

| 配置 | 硬件 | 说明 |
|------|------|------|
| `DSV4_PROFILE=4layer`（默认） | 单机 8 GPU | 4-layer GRPO smoke |
| `DSV4_PROFILE=flash` | 2 节点 16 GPU | 43-layer 全模型 GRPO / pretrain |

加载预训练 checkpoint → **GRPO policy loss** 更新几乎全部权重（MoE router gate / e-score bias 冻结）。默认 `DEBUG_TRAIN_ONLY=1`（预生成 `fake_rollout.pt`，无 SGLang / Ray）。

**Finetune 默认 batch**（4-layer 与 flash 相同，由 `dsv4_finetune_common.sh` 设置）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `GBS` | `256` | 须等于 rollout 样本数（32 prompts × 8） |
| `SEQ_LEN` | `4096` | Megatron `--seq-length` |
| `MBS` | `1` | micro-batch |

Pretrain（`DSV4_MODE=pretrain`）默认 `GBS=8`、`SEQ_LEN=2048`，与 finetune 不同。

---

## 前置条件

| 项目 | 说明 |
|------|------|
| 硬件 | 8× MI300X/MI308X（flash 需 2 节点各 8 卡），宿主机可访问 `/dev/kfd` |
| Docker | 已安装，当前用户可运行 |
| 基础镜像 | `lumen/tests:latest`（构建 DSV4 镜像用） |
| AIter | 训练容器内 `/workspace/aiter`，提供 DSV4 MHC Triton API |

默认路径由 `dsv4_paths.sh` 解析：

```text
WORKSPACE_ROOT=..
DATA_ROOT=                     # /nfs/data/$USER → /mnt/data/$USER → ${WORKSPACE_ROOT}/dsv4-data
MODEL_DIR=${DATA_ROOT}/models
LOG_DIR=${DATA_ROOT}/logs
BOOTSTRAP_DIR=${DATA_ROOT}/lumen-dsv4-bootstrap
```

---

## 目录说明

```text
examples/dsv4/
├── dsv4_launch.sh                      # 统一 Docker launcher（推荐入口）
├── launch_dsv4_2node.sh                # Flash 2-node 一键拉起（DSV4_MODE=finetune|pretrain）
├── run_dsv4_inner.sh                     # 容器内 GRPO finetune torchrun
├── run_dsv4_pretrain_inner.sh            # 容器内 pretrain/profile torchrun
├── dsv4_megatron_args.sh                 # 4layer/flash 模型参数
├── dsv4_finetune_common.sh               # batch / ckpt / rollout helpers
├── megatron_patch/                       # ROCm Megatron 模块化 patch
├── prepare_dsv4_checkpoint.py            # HF → BF16 → torch_dist
├── finetune_dsv4_megatron.py             # GRPO 主程序
├── pretrain_dsv4_megatron.py             # Pretrain 主程序（含 profiler hook）
└── …
```

模型实现：`lumen/models/dsv4/`（`compressor.py`、`hyper_connection.py`、`megatron/`）；运维工具：`lumen/tools/dsv4/`。

---

## 快速开始

### 1. 构建镜像

```bash
cd ~/Lumen
bash examples/dsv4/build_dsv4_lumen_image.sh
```

### 2. 4-layer GRPO finetune（单机 8 卡）

```bash
# 首次：自动 prepare checkpoint + GSM8K + fake rollout
bash examples/dsv4/dsv4_launch.sh

# 已有 checkpoint + rollout
SKIP_PREPARE=1 DSV4_HC_MULT=4 bash examples/dsv4/dsv4_launch.sh
```

日志：`${LOG_DIR}/lumen_dsv4_4layer_finetune_*.log`

成功标志：

```text
=== [done] Lumen DSV4 4layer native GRPO finetune completed ===
```

### 3. Flash 全模型 GRPO finetune（2×8，16 GPU）

**推荐：head 一键拉起**

```bash
MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-host> \
MODEL_DIR=/data1/${USER}/models \
WORKER_MODEL_DIR=/mnt/nvme0n1/${USER}/models \
SKIP_PREPARE=1 DSV4_HC_MULT=4 \
  bash examples/dsv4/launch_dsv4_2node.sh
```

默认 `GBS=256`、`SEQ_LEN=4096`、`NUM_ROLLOUT=10`（见 launch 脚本）。checkpoint 建议放各节点本地 NVMe，rollout 仍用 NFS `${DATA_ROOT}/models/fake_rollout.pt`。

**手动两节点**

```bash
# head
NODE_RANK=0 MASTER_ADDR=<head-ip> MODEL_DIR=/data1/${USER}/models \
  DSV4_MODE=finetune DSV4_PROFILE=flash SKIP_PREPARE=1 DSV4_HC_MULT=4 \
  bash examples/dsv4/dsv4_launch.sh

# worker（head preflight 通过后再启动）
NODE_RANK=1 MASTER_ADDR=<head-ip> MODEL_DIR=/mnt/nvme0n1/${USER}/models \
  DSV4_MODE=finetune DSV4_PROFILE=flash SKIP_PREPARE=1 DSV4_HC_MULT=4 \
  bash examples/dsv4/dsv4_launch.sh
```

### 4. Flash 43L pretrain smoke

```bash
LOAD_CKPT=1 SKIP_PREPARE=1 TRAIN_ITERS=10 \
MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-host> \
MODEL_DIR=/data1/${USER}/models WORKER_MODEL_DIR=/mnt/nvme0n1/${USER}/models \
DSV4_MODE=pretrain bash examples/dsv4/launch_dsv4_2node.sh
```

详见 [runbook.md](./runbook.md) §5。

---

## 常用环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `DSV4_MODE` | `finetune` | `finetune` / `pretrain` / `profile` |
| `DSV4_PROFILE` | `4layer` | `4layer` 或 `flash` |
| `NUM_ROLLOUT` | `10` | GRPO 训练步数 |
| `GBS` | `256`（finetune）/ `8`（pretrain） | global batch |
| `SEQ_LEN` | `4096`（finetune）/ `2048`（pretrain） | 训练序列长度 |
| `MBS` | `1` | micro-batch |
| `DSV4_HC_MULT` | `4` | 须与 checkpoint 目录 `_torch_dist_hc{N}` 一致 |
| `SKIP_PREPARE` | `0` | `1` = 跳过 HF→torch_dist |
| `DEBUG_TRAIN_ONLY` | `1` | 必须为 1（无 SGLang live rollout） |
| `V4_INDEXER_IMPL` | `aiter` | DSA indexer |
| `V4_SPARSE_MLA_BACKEND` | `triton` | sparse MLA |
| MHC | AIter | Hyper-Connection 直接调用 AIter DSV4 fused API |
| `OPTIMIZER_OFFLOAD_FRACTION` | `0.75` | Flash 全模型 CPU Adam offload |
| `DSV4_ENABLE_EXPERT_BIAS` | `0` | converted ckpt 常缺 expert_bias shard |

---

## 架构

```text
dsv4_launch.sh (host)
  └─ docker run lumen/dsv4-lumen:mi308x
       ├─ run_dsv4_inner.sh              (DSV4_MODE=finetune)
       │    ├─ setup_container_env.sh
       │    ├─ prepare_dsv4_checkpoint.py  (可选)
       │    ├─ gen_fake_rollout_data.prepare()  (rank0)
       │    └─ torchrun finetune_dsv4_megatron.py
       └─ run_dsv4_pretrain_inner.sh     (DSV4_MODE=pretrain|profile)
            └─ torchrun pretrain_dsv4_megatron.py
                 └─ lumen.models.dsv4.megatron.spec.get_dsv4_spec
```

双节点操作细节见 [runbook.md](./runbook.md)。

`tile_kernels` 仍仅供 `lumen/ops/dsv4/qat.py` 的 FP8 QAT quant kernels 使用，不再参与 Hyper-Connection。
