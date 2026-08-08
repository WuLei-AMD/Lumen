#!/usr/bin/env bash
# Host wrapper: tune DSV4 BF16 GEMM on MI308X via docker, then merge into consolidated CSV.
#
# Usage:
#   bash examples/dsv4/run_tune_dsv4_bf16_gemm_mi308x.sh
#   TUNE_LIBTYPE=asm,triton,skinny,opus bash examples/dsv4/run_tune_dsv4_bf16_gemm_mi308x.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"

IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"
GEMM_TUNE_DIR="${LUMEN_DIR}/examples/dsv4/.gemm_tune"
LOGFILE="${LOG_DIR}/dsv4_bf16_gemm_tune_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}" "${GEMM_TUNE_DIR}"

echo "[run_tune] log=${LOGFILE}"

docker run --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --ipc=host \
    -v "${LUMEN_DIR}:/workspace/Lumen" \
    -e LUMEN_DIR=/workspace/Lumen \
    -e AITER_ROOT=/workspace/Lumen/third_party/aiter \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    -e TUNE_INPUT="${TUNE_INPUT:-/workspace/Lumen/examples/dsv4/.gemm_tune/dsv4_bf16_moe_bucket_untuned.csv}" \
    -e TUNE_OUTPUT="${TUNE_OUTPUT:-/workspace/Lumen/examples/dsv4/.gemm_tune/dsv4_bf16_tuned_gfx942_mi308x.csv}" \
    -e TUNE_LIBTYPE="${TUNE_LIBTYPE:-asm,triton,skinny,opus}" \
    -e TUNE_BATCH="${TUNE_BATCH:-10}" \
    "${IMAGE}" \
    bash /workspace/Lumen/examples/dsv4/tune_dsv4_bf16_gemm_mi308x.sh \
    2>&1 | tee "${LOGFILE}"

python3 "${LUMEN_DIR}/examples/dsv4/merge_dsv4_bf16_gemm_mi308x.py"

echo "[run_tune] merged CSV -> ${LUMEN_DIR}/examples/dsv4/configs/dsv4_bf16_tuned_gemm_mi308x.csv"
echo "[run_tune] tune log -> ${LOGFILE}"
