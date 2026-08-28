#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME=${IMAGE_NAME:-zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260828}
COMMAND=${COMMAND:-"bash run_qwen3_30b_a3b_megatron.sh"}

docker run --rm --init \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --network=host \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --shm-size=64G \
    --volume "${REPO_ROOT}/lumen/models/megatron.py:/workspace/Lumen/lumen/models/megatron.py" \
    --volume "${REPO_ROOT}/lumen/models/qwen3_30b_a3b/fsdp:/workspace/Lumen/lumen/models/qwen3_30b_a3b/fsdp" \
    --volume "${REPO_ROOT}/examples:/workspace/Lumen/examples" \
    --volume /mnt/m2m_nobackup/danyzhan/qwen3-30b-a3b:/nobackup \
    --volume lumen-qwen3-triton-cache:/root/.triton \
    --workdir /workspace/Lumen/examples/qwen3-30b-a3b \
    --env PYTHONPATH=/workspace/Lumen:/workspace/Lumen/third_party/aiter:/workspace/Megatron-LM \
    --env MOE_IMPL="${MOE_IMPL:-sequential}" \
    --env TRAIN_STEPS="${TRAIN_STEPS:-20}" \
    --env SEQ_LEN="${SEQ_LEN:-4096}" \
    --env MBS="${MBS:-1}" \
    --env GBS="${GBS:-8}" \
    --env MODEL_PATH="${MODEL_PATH:-}" \
    --env DATA_PATH="${DATA_PATH:-}" \
    --env MEGATRON_LOAD_PATH="${MEGATRON_LOAD_PATH:-}" \
    --env LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2}" \
    --env MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" \
    --env MASTER_PORT="${MASTER_PORT:-29500}" \
    --env RUN_SUFFIX="${RUN_SUFFIX:-}" \
    --env EXPERT_BACKEND="${EXPERT_BACKEND:-te_grouped}" \
    --env SONIC_MOE_GEMM_BACKEND="${SONIC_MOE_GEMM_BACKEND:-triton}" \
    --env NVTE_USE_CUTLASS_GROUPED_GEMM="${NVTE_USE_CUTLASS_GROUPED_GEMM:-0}" \
    --env NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK="${NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK:-0}" \
    --env NVTE_USE_HIPBLASLT="${NVTE_USE_HIPBLASLT:-}" \
    --env NVTE_USE_HIPKITTENS_GEMM="${NVTE_USE_HIPKITTENS_GEMM:-}" \
    --env TE_HIPBLASLT_TUNING_LOAD="${TE_HIPBLASLT_TUNING_LOAD:-}" \
    --env HIPBLASLT_TUNING_FILE="${HIPBLASLT_TUNING_FILE:-}" \
    --env HIPBLASLT_TUNING_OVERRIDE_FILE="${HIPBLASLT_TUNING_OVERRIDE_FILE:-}" \
    "${IMAGE_NAME}" \
    bash -lc "${COMMAND}"
