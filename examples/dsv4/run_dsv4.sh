#!/usr/bin/env bash
# run_dsv4.sh — DSV4 GRPO full finetune Docker launcher.
#
# Profiles:
#   DSV4_PROFILE=4layer   — 4-layer, single-node 8 GPU (default)
#   DSV4_PROFILE=flash    — 43-layer full model, 2-node 16 GPU
#
# Examples:
#   bash examples/dsv4/run_dsv4.sh
#   DSV4_PROFILE=flash NODE_RANK=0 MASTER_ADDR=<head-ip> bash examples/dsv4/run_dsv4.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"
# shellcheck source=examples/dsv4/dsv4_docker_common.sh
source "${SCRIPT_DIR}/dsv4_docker_common.sh"

DSV4_PROFILE="${DSV4_PROFILE:-4layer}"
IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"

if [[ "${DSV4_PROFILE}" == "4layer" ]]; then
    MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8-4layer}"
    NNODES=1
    NPROC_PER_NODE=8
    NODE_RANK=0
    DSV4_HC_MULT="${DSV4_HC_MULT:-4}"
elif [[ "${DSV4_PROFILE}" == "flash" ]]; then
    MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8}"
    NNODES="${NNODES:-2}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    NODE_RANK="${NODE_RANK:-0}"
    MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to head node IP}"
    MASTER_PORT="${MASTER_PORT:-29500}"
    DSV4_HC_MULT="${DSV4_HC_MULT:-4}"
else
    echo "[ERROR] Unknown DSV4_PROFILE=${DSV4_PROFILE} (expected 4layer or flash)"
    exit 1
fi

DATA_DIR="${DATA_DIR:-${DATA_ROOT}/datasets}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY:-1}"
GBS="${GBS:-256}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-180}"
V4_SPARSE_MLA_BACKEND="${V4_SPARSE_MLA_BACKEND:-triton}"
MHC_BACKEND="${MHC_BACKEND:-triton}"
V4_INDEXER_IMPL="${V4_INDEXER_IMPL:-aiter}"
V4_INDEXER_BLOCK_N="${V4_INDEXER_BLOCK_N:-64}"
V4_INDEXER_NUM_STAGES="${V4_INDEXER_NUM_STAGES:-1}"
LUMEN_DSV4_LINEAR_FP8="${LUMEN_DSV4_LINEAR_FP8:-0}"

# shellcheck source=examples/dsv4/dsv4_megatron_args.sh
source "${SCRIPT_DIR}/dsv4_megatron_args.sh"

if [[ "${DSV4_PROFILE}" == "4layer" ]]; then
    TORCH_DIST="${MODEL_DIR}/${MODEL_NAME}_torch_dist_hc${DSV4_HC_MULT}"
    LOGFILE="${LOG_DIR}/lumen_dsv4_4layer_finetune_$(date +%Y%m%d_%H%M%S).log"
    CONTAINER_NAME="lumen-dsv4-finetune"
else
    TORCH_DIST="${MODEL_DIR}/${MODEL_NAME}_torch_dist"
    LOGFILE="${LOG_DIR}/lumen_dsv4_flash_finetune_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
    CONTAINER_NAME="lumen-dsv4-flash-finetune-node${NODE_RANK}"
fi

dsv4_docker_bootstrap_setup
dsv4_docker_ensure_image
dsv4_docker_check_rocm
dsv4_docker_mkdirs

if [[ "${DSV4_PROFILE}" == "flash" && "${NNODES}" -gt 1 ]]; then
    # shellcheck source=examples/dsv4/preflight_dsv4_flash_multinode.sh
    source "${SCRIPT_DIR}/preflight_dsv4_flash_multinode.sh"
    preflight_dsv4_multinode
fi

echo "════════════════════════════════════════════════"
echo "  Lumen DSV4 ${DSV4_PROFILE} GRPO full finetune"
echo "  Image     : ${IMAGE}"
echo "  Workspace : ${WORKSPACE_ROOT}  (data: ${DATA_ROOT})"
if [[ "${DSV4_PROFILE}" == "flash" ]]; then
    echo "  Nodes     : ${NNODES}×${NPROC_PER_NODE}  node_rank=${NODE_RANK}"
    echo "  Master    : ${MASTER_ADDR}:${MASTER_PORT}"
    echo "  Parallel  : TP4 PP4 EP4 (11+11+11+10)"
fi
echo "  Rollouts  : ${NUM_ROLLOUT}"
echo "  Mode      : native torchrun GRPO (debug-train-only=${DEBUG_TRAIN_ONLY})"
echo "  Batch     : GBS=${GBS} MBS=${MBS} seq_len=${SEQ_LEN}"
echo "  HC mult   : ${DSV4_HC_MULT} (MHC_BACKEND=${MHC_BACKEND})"
echo "  SparseMLA : ${V4_SPARSE_MLA_BACKEND}"
echo "  Ckpt      : ${TORCH_DIST}"
echo "  Log       : ${LOGFILE}"
echo "════════════════════════════════════════════════"

dsv4_docker_build_mounts

DOCKER_ENV=(
    -e LUMEN_DIR=/workspace/Lumen
    -e DSV4_PROFILE="${DSV4_PROFILE}"
    -e LUMEN_DSV4_NATIVE_FINETUNE=1
    -e MODEL_DIR=/root/models
    -e DATA_DIR=/root/datasets
    -e MODEL_NAME="${MODEL_NAME}"
    -e DSV4_HC_MULT="${DSV4_HC_MULT}"
    -e SKIP_PREPARE="${SKIP_PREPARE}"
    -e NUM_ROLLOUT="${NUM_ROLLOUT}"
    -e DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY}"
    -e GBS="${GBS}"
    -e MBS="${MBS}"
    -e SEQ_LEN="${SEQ_LEN}"
    -e LUMEN_DSV4_LINEAR_FP8="${LUMEN_DSV4_LINEAR_FP8}"
)
if [[ "${DSV4_PROFILE}" == "flash" ]]; then
    dsv4_docker_append_multinode_env
fi
dsv4_docker_append_kernel_env
dsv4_docker_append_rocm_env
dsv4_docker_append_bootstrap_env

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

if [[ "${DSV4_PROFILE}" == "flash" ]]; then
    dsv4_docker_run_opts_multinode
else
    dsv4_docker_run_opts_single
fi

"${DOCKER_RUN[@]}" \
    "${DOCKER_MOUNTS[@]}" \
    "${DOCKER_ENV[@]}" \
    "${IMAGE}" \
    bash /workspace/Lumen/examples/dsv4/run_dsv4_inner.sh \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "Log saved to: ${LOGFILE}"
