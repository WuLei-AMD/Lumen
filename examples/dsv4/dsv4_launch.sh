#!/usr/bin/env bash
# Unified DSV4 Docker launcher (outer wrapper).
#
# DSV4_MODE=pretrain|finetune|profile   (default: finetune)
# DSV4_PROFILE=4layer|flash             (default: 4layer)
#
# Examples:
#   bash examples/dsv4/dsv4_launch.sh
#   DSV4_MODE=pretrain DSV4_PROFILE=4layer bash examples/dsv4/dsv4_launch.sh
#   DSV4_MODE=finetune DSV4_PROFILE=flash NODE_RANK=0 MASTER_ADDR=<ip> bash examples/dsv4/dsv4_launch.sh
#   DSV4_MODE=profile bash examples/dsv4/dsv4_launch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"
# shellcheck source=examples/dsv4/dsv4_docker_common.sh
source "${SCRIPT_DIR}/dsv4_docker_common.sh"

DSV4_MODE="${DSV4_MODE:-finetune}"
DSV4_PROFILE="${DSV4_PROFILE:-4layer}"
IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"

if [[ "${DSV4_MODE}" == "profile" && "${DSV4_PROFILE}" != "4layer" ]]; then
    echo "[launch] ERROR: profile mode only supports DSV4_PROFILE=4layer"
    exit 1
fi

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
    MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to head node IP for flash profile}"
    MASTER_PORT="${MASTER_PORT:-29500}"
    DSV4_HC_MULT="${DSV4_HC_MULT:-4}"
else
    echo "[launch] ERROR: unknown DSV4_PROFILE=${DSV4_PROFILE} (expected 4layer or flash)"
    exit 1
fi

# shellcheck source=examples/dsv4/dsv4_megatron_args.sh
source "${SCRIPT_DIR}/dsv4_megatron_args.sh"

DATA_DIR="${DATA_DIR:-${DATA_ROOT}/datasets}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
LUMEN_DSV4_LINEAR_FP8="${LUMEN_DSV4_LINEAR_FP8:-0}"
LUMEN_DSV4_FP8_SCALING="${LUMEN_DSV4_FP8_SCALING:-blockwise}"
LUMEN_DSV4_MOE_MORI="${LUMEN_DSV4_MOE_MORI:-0}"
MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-0}"

INNER_SCRIPT="/workspace/Lumen/examples/dsv4/run_dsv4_inner.sh"
CONTAINER_NAME="lumen-dsv4-${DSV4_MODE}"
DOCKER_ENV=(
    -e LUMEN_DIR=/workspace/Lumen
    -e DSV4_PROFILE="${DSV4_PROFILE}"
    -e MODEL_DIR=/root/models
    -e MODEL_NAME="${MODEL_NAME}"
    -e DSV4_HC_MULT="${DSV4_HC_MULT}"
    -e SKIP_PREPARE="${SKIP_PREPARE}"
)

case "${DSV4_MODE}" in
    finetune)
        # shellcheck source=examples/dsv4/dsv4_finetune_common.sh
        source "${SCRIPT_DIR}/dsv4_finetune_common.sh"
        NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
        DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY:-1}"
        GBS="${GBS:-256}"
        OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
        DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-180}"
        dsv4_apply_finetune_batch_defaults
        dsv4_resolve_fake_rollout_path
        dsv4_host_resolve_finetune_ckpt "${MODEL_NAME}" "${DSV4_HC_MULT}"

        if [[ "${DSV4_PROFILE}" == "4layer" ]]; then
            LOGFILE="${LOG_DIR}/lumen_dsv4_4layer_finetune_$(date +%Y%m%d_%H%M%S).log"
            CONTAINER_NAME="lumen-dsv4-finetune"
            CKPT_BANNER="${CKPT_HOST}"
        else
            LOGFILE="${LOG_DIR}/lumen_dsv4_flash_finetune_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
            CONTAINER_NAME="lumen-dsv4-flash-finetune-node${NODE_RANK}"
            CKPT_BANNER="${MODEL_DIR}/${MODEL_NAME}_torch_dist"
        fi

        DOCKER_ENV+=(
            -e LUMEN_DSV4_NATIVE_FINETUNE=1
            -e LUMEN_DSV4_PRETRAIN=1
            -e DATA_DIR=/root/datasets
            -e DATA_ROOT="${DATA_ROOT}"
            -e NUM_ROLLOUT="${NUM_ROLLOUT}"
            -e DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY}"
            -e GBS="${GBS}"
            -e MBS="${MBS}"
            -e SEQ_LEN="${SEQ_LEN}"
            -e DSV4_KEEP_GBS="${DSV4_KEEP_GBS:-0}"
            -e DSV4_KEEP_SEQ_LEN="${DSV4_KEEP_SEQ_LEN:-0}"
            -e ROLLOUT_N_PROMPTS="${ROLLOUT_N_PROMPTS:-32}"
            -e ROLLOUT_N_PER_PROMPT="${ROLLOUT_N_PER_PROMPT:-8}"
            -e SMOKE_LEGACY_FAKE_ROLLOUT="${SMOKE_LEGACY_FAKE_ROLLOUT:-0}"
            -e FAKE_ROLLOUT_DATA="${FAKE_ROLLOUT_DATA}"
            -e LUMEN_DSV4_LINEAR_FP8="${LUMEN_DSV4_LINEAR_FP8}"
        )
        INNER_SCRIPT="/workspace/Lumen/examples/dsv4/run_dsv4_inner.sh"
        ;;
    pretrain)
        TRAIN_ITERS="${TRAIN_ITERS:-10}"
        LOAD_CKPT="${LOAD_CKPT:-0}"
        EVAL_ITERS="${EVAL_ITERS:-1}"
        OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
        DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-180}"

        if [[ "${DSV4_PROFILE}" == "4layer" ]]; then
            LOGFILE="${LOG_DIR}/lumen_dsv4_4layer_pretrain_$(date +%Y%m%d_%H%M%S).log"
            CONTAINER_NAME="lumen-dsv4-pretrain"
            CKPT_BANNER="${MODEL_DIR}/${MODEL_NAME}_torch_dist_hc${DSV4_HC_MULT}"
        else
            LOGFILE="${LOG_DIR}/lumen_dsv4_flash_pretrain_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
            CONTAINER_NAME="lumen-dsv4-full-node${NODE_RANK}"
            CKPT_BANNER="${MODEL_DIR}/${MODEL_NAME}_torch_dist"
        fi

        DOCKER_ENV+=(
            -e LUMEN_DSV4_PRETRAIN=1
            -e TRAIN_ITERS="${TRAIN_ITERS}"
            -e GBS="${GBS}"
            -e MBS="${MBS}"
            -e SEQ_LEN="${SEQ_LEN}"
            -e LOAD_CKPT="${LOAD_CKPT}"
            -e EVAL_ITERS="${EVAL_ITERS}"
            -e LUMEN_DSV4_LINEAR_FP8="${LUMEN_DSV4_LINEAR_FP8}"
            -e LUMEN_DSV4_FP8_SCALING="${LUMEN_DSV4_FP8_SCALING}"
            -e LUMEN_DSV4_MOE_MORI="${LUMEN_DSV4_MOE_MORI}"
        )
        INNER_SCRIPT="/workspace/Lumen/examples/dsv4/run_dsv4_pretrain_inner.sh"
        ;;
    profile)
        RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
        mkdir -p "${RESULTS_DIR}"
        LUMEN_RUNTIME_DIR="${LUMEN_RUNTIME_DIR:-${LOG_DIR}/lumen-dsv4-runtime}"
        CONTAINER_TMPDIR="${CONTAINER_TMPDIR:-/dev/shm/lumen-dsv4-tmp}"
        mkdir -p "${LUMEN_RUNTIME_DIR}"
        TRAIN_ITERS="${TRAIN_ITERS:-5}"
        LOAD_CKPT="${LOAD_CKPT:-1}"
        SKIP_PREPARE="${SKIP_PREPARE:-1}"
        LUMEN_PROF_START="${LUMEN_PROF_START:-3}"
        LUMEN_PROF_END="${LUMEN_PROF_END:-5}"
        LUMEN_PROF_STOP_AFTER="${LUMEN_PROF_STOP_AFTER:-${LUMEN_PROF_END}}"

        LOGFILE="${LOG_DIR}/lumen_dsv4_4layer_profile_$(date +%Y%m%d_%H%M%S).log"
        CONTAINER_NAME="lumen-dsv4-profile"
        CKPT_BANNER="${MODEL_DIR}/${MODEL_NAME}_torch_dist_hc${DSV4_HC_MULT}"

        DOCKER_ENV+=(
            -e LUMEN_DSV4_PRETRAIN=1
            -e TRAIN_ITERS="${TRAIN_ITERS}"
            -e GBS="${GBS}"
            -e MBS="${MBS}"
            -e SEQ_LEN="${SEQ_LEN}"
            -e LOAD_CKPT="${LOAD_CKPT}"
            -e EVAL_ITERS=0
            -e LUMEN_PROF_START="${LUMEN_PROF_START}"
            -e LUMEN_PROF_END="${LUMEN_PROF_END}"
            -e LUMEN_PROF_STOP_AFTER="${LUMEN_PROF_STOP_AFTER}"
            -e LUMEN_PROF_OUTPUT="/workspace/Lumen/examples/dsv4/results/dsv4_4layer_profile.txt"
            -e LUMEN_PROF_XLSX="/workspace/Lumen/examples/dsv4/results/dsv4_4layer_operator_breakdown.xlsx"
            -e LUMEN_PROF_SHAPES="${LUMEN_PROF_SHAPES:-0}"
            -e LUMEN_PROF_TRACE="${LUMEN_PROF_TRACE:-}"
            -e RESULTS_DIR=/workspace/Lumen/examples/dsv4/results
            -e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
            -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens14np0}"
            -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-ens14np0}}"
            -e MEGATRON_PATH="${MEGATRON_PATH}"
            -e TL_DISABLE_OUT_OF_BOUND_WARNING="${TL_DISABLE_OUT_OF_BOUND_WARNING:-1}"
            -e TMPDIR="${CONTAINER_TMPDIR}"
        )
        INNER_SCRIPT="/workspace/Lumen/examples/dsv4/run_dsv4_pretrain_inner.sh"
        export DSV4_MODE=profile
        ;;
    *)
        echo "[launch] ERROR: unknown DSV4_MODE=${DSV4_MODE} (expected pretrain, finetune, or profile)"
        exit 1
        ;;
esac

dsv4_docker_prepare_host

if [[ "${DSV4_PROFILE}" == "flash" && "${NNODES}" -gt 1 ]]; then
    # shellcheck source=examples/dsv4/preflight_dsv4_flash_multinode.sh
    source "${SCRIPT_DIR}/preflight_dsv4_flash_multinode.sh"
    preflight_dsv4_multinode
fi

echo "════════════════════════════════════════════════"
echo "  Lumen DSV4 ${DSV4_PROFILE} ${DSV4_MODE}"
echo "  Image     : ${IMAGE}"
echo "  Workspace : ${WORKSPACE_ROOT}  (data: ${DATA_ROOT})"
if [[ "${DSV4_PROFILE}" == "flash" ]]; then
    echo "  Nodes     : ${NNODES}×${NPROC_PER_NODE}  node_rank=${NODE_RANK}"
    echo "  Master    : ${MASTER_ADDR}:${MASTER_PORT}"
fi
if [[ "${DSV4_MODE}" == "finetune" ]]; then
    echo "  Rollouts  : ${NUM_ROLLOUT}"
    echo "  Batch     : GBS=${GBS} MBS=${MBS} seq_len=${SEQ_LEN}"
    echo "  Rollout   : ${FAKE_ROLLOUT_DATA:-n/a}"
elif [[ "${DSV4_MODE}" == "pretrain" ]]; then
    echo "  Steps     : ${TRAIN_ITERS}"
    echo "  Batch     : GBS=${GBS} MBS=${MBS} seq_len=${SEQ_LEN}"
    echo "  Load ckpt : LOAD_CKPT=${LOAD_CKPT}"
elif [[ "${DSV4_MODE}" == "profile" ]]; then
    echo "  Profile   : steps ${LUMEN_PROF_START}-${LUMEN_PROF_END}"
    echo "  Results   : ${RESULTS_DIR}"
fi
echo "  HC mult   : ${DSV4_HC_MULT}"
echo "  Indexer   : ${V4_INDEXER_IMPL}"
echo "  SparseMLA : ${V4_SPARSE_MLA_BACKEND}"
if [[ "${DSV4_MODE}" == "finetune" ]]; then
    dsv4_print_gemm_env
fi
echo "  Ckpt      : ${CKPT_BANNER}"
echo "  Log       : ${LOGFILE}"
echo "════════════════════════════════════════════════"

dsv4_docker_build_mounts
if [[ "${DSV4_MODE}" == "profile" ]]; then
    DOCKER_MOUNTS+=(
        -v "${LUMEN_RUNTIME_DIR}:/opt/dsv4-runtime"
        -v "${RESULTS_DIR}:/workspace/Lumen/examples/dsv4/results"
    )
fi

if [[ -d "${MILES_DIR:-}" ]]; then
    DOCKER_ENV+=(-e MILES_DIR=/workspace/miles)
fi
if [[ "${MORI_ENABLE_SDMA}" == "1" ]]; then
    DOCKER_ENV+=(-e MORI_ENABLE_SDMA=1)
fi
if [[ "${DSV4_PROFILE}" == "flash" ]]; then
    dsv4_docker_append_multinode_env
fi
dsv4_docker_append_kernel_env
dsv4_docker_append_gemm_env
dsv4_docker_append_rocm_env
dsv4_docker_append_bootstrap_env
if [[ "${DSV4_MODE}" == "finetune" && "${DSV4_PROFILE}" == "flash" ]]; then
    dsv4_docker_append_miles_ray_env
fi

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
if [[ "${DSV4_PROFILE}" == "flash" ]]; then
    dsv4_docker_run_opts_multinode
else
    dsv4_docker_run_opts_single
fi

"${DOCKER_RUN[@]}" \
    "${DOCKER_DEVICES[@]}" \
    "${DOCKER_MOUNTS[@]}" \
    "${DOCKER_ENV[@]}" \
    -e DSV4_MODE="${DSV4_MODE}" \
    "${IMAGE}" \
    bash "${INNER_SCRIPT}" \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "Log saved to: ${LOGFILE}"
