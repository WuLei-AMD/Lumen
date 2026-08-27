#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME=${IMAGE_NAME:-lumen/qwen3-30b-a3b:latest}
COMMAND=${COMMAND:-"bash run_pretrain.sh"}

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
    --volume "${REPO_ROOT}:/workspace/Lumen" \
    --workdir /workspace/Lumen/examples/qwen3-30b-a3b \
    --env PYTHONPATH=/workspace/Lumen:/workspace/Lumen/third_party/aiter:/workspace/Megatron-LM \
    --env MOE_IMPL="${MOE_IMPL:-sequential}" \
    --env TRAIN_STEPS="${TRAIN_STEPS:-20}" \
    --env SEQ_LEN="${SEQ_LEN:-4096}" \
    --env MBS="${MBS:-1}" \
    --env GBS="${GBS:-8}" \
    "${IMAGE_NAME}" \
    bash -lc "${COMMAND}"
