#!/usr/bin/env bash
# Run TileLang vs Triton sparse-MLA kernel compare on 1 GPU in the Lumen image.
set -euo pipefail

LUMEN_DIR="${LUMEN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MILES_DIR="${MILES_DIR:-$(cd "${LUMEN_DIR}/../miles" 2>/dev/null && pwd || true)}"
IMAGE="${IMAGE:-zhangdanyangamd/lumen:dsv4-flash-308x-finetune}"
AITER_DIR="${AITER_DIR:-${LUMEN_DIR}/third_party/aiter}"
GPU="${GPU:-0}"

if [[ ! -d "${AITER_DIR}/aiter" ]]; then
    echo "[ERROR] AITER_DIR missing aiter package: ${AITER_DIR}"
    exit 1
fi

MILES_MOUNT=()
if [[ -n "${MILES_DIR}" && -d "${MILES_DIR}/miles_plugins" ]]; then
    MILES_MOUNT=(-v "${MILES_DIR}:/workspace/miles")
fi

docker rm -f lumen-sparse-mla-cmp 2>/dev/null || true
docker run --rm \
    --name lumen-sparse-mla-cmp \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --shm-size=32g \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES="${GPU}" \
    -e CUDA_VISIBLE_DEVICES="${GPU}" \
    -e AITER_DIR=/workspace/aiter \
    -e LUMEN_DIR=/workspace/Lumen \
    -e TILELANG_DIR=/opt/dsv4-runtime/tilelang \
    -e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}" \
    -v "${LUMEN_DIR}:/workspace/Lumen" \
    -v "${AITER_DIR}:/workspace/aiter" \
    "${MILES_MOUNT[@]}" \
    "${IMAGE}" \
    bash -c 'export PYTHONPATH="/workspace/Lumen:/workspace/miles:/workspace/aiter:/opt/dsv4-runtime/tilelang:/opt/dsv4-runtime/site-packages:${PYTHONPATH:-}"; python /workspace/Lumen/examples/dsv4/tools/compare_sparse_mla_kernels.py "$@"' -- "$@"
