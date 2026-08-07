#!/usr/bin/env bash
# Launch DSV4 Flash 2-node training from the head node with identical env on both ranks.
#
# Usage (on head node):
#   cd ~/Lumen
#   MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-host> \
#     bash examples/dsv4/launch_dsv4_2node.sh
#
# Modes (DSV4_MODE):
#   finetune (default) — GRPO full finetune, GBS=256 SEQ_LEN=4096 NUM_ROLLOUT=10
#   pretrain           — Megatron mock-data pretrain, GBS=8 SEQ_LEN=2048
#
# Finetune optional overrides:
#   SKIP_PREPARE=1 GBS=256 NUM_ROLLOUT=10 DSV4_HC_MULT=4 IMAGE=lumen/dsv4-lumen:mi308x
#   WORKER_MODEL_DIR=/mnt/nvme0n1/${USER}/models
#   Smoke: GBS=8 DSV4_KEEP_GBS=1 SEQ_LEN=512 DSV4_KEEP_SEQ_LEN=1 NUM_ROLLOUT=2 \
#     ROLLOUT_N_PROMPTS=1 ROLLOUT_N_PER_PROMPT=8 SMOKE_LEGACY_FAKE_ROLLOUT=1
#   Bisect: TP=4 PP=2 EP=1 DECODER_FIRST_PP_LAYERS=22 DECODER_LAST_PP_LAYERS=21
#
# Pretrain optional overrides:
#   LOAD_CKPT=1 SKIP_PREPARE=1 TRAIN_ITERS=10 EVAL_ITERS=0 RERUN_MODE=disabled

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"

DSV4_MODE="${DSV4_MODE:-finetune}"
DSV4_PROFILE="${DSV4_PROFILE:-flash}"
if [[ "${DSV4_MODE}" != "finetune" && "${DSV4_MODE}" != "pretrain" ]]; then
    echo "[launch] ERROR: DSV4_MODE=${DSV4_MODE} (expected finetune or pretrain)" >&2
    exit 1
fi
if [[ "${DSV4_PROFILE}" != "flash" ]]; then
    echo "[launch] ERROR: 2-node launcher requires DSV4_PROFILE=flash (got ${DSV4_PROFILE})" >&2
    exit 1
fi

MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to head node IP}"
WORKER_ADDR="${WORKER_ADDR:-}"
WORKER_SSH="${WORKER_SSH:-}"
if [[ -z "${WORKER_SSH}" && -n "${WORKER_ADDR}" ]]; then
    WORKER_SSH="${USER}@${WORKER_ADDR}"
fi
WORKER_SSH="${WORKER_SSH:?Set WORKER_SSH (e.g. ${USER}@worker-host) or WORKER_ADDR}"
SSH_KEY="${SSH_KEY:-}"
PREFLIGHT_ID="$(date +%Y%m%d_%H%M%S)"
WORKER_MODEL_DIR="${WORKER_MODEL_DIR:-${MODEL_DIR}}"

if [[ "${DSV4_MODE}" == "finetune" ]]; then
    CONTAINER_PREFIX="lumen-dsv4-flash-finetune-node"
    LOG_TAG="finetune"
    TRAIN_LOG_GLOB="lumen_dsv4_flash_finetune_node"
else
    CONTAINER_PREFIX="lumen-dsv4-full-node"
    LOG_TAG="pretrain"
    TRAIN_LOG_GLOB="lumen_dsv4_flash_pretrain_node"
fi

COMMON_ENV=(
    "DSV4_PROFILE=${DSV4_PROFILE}"
    "PREFLIGHT_ID=${PREFLIGHT_ID}"
    "MASTER_ADDR=${MASTER_ADDR}"
    "AITER_DIR=${AITER_DIR}"
    "MASTER_PORT=${MASTER_PORT:-29500}"
    "MODEL_DIR=${MODEL_DIR}"
    "DATA_ROOT=${DATA_ROOT}"
    "LOG_DIR=${LOG_DIR}"
    "SKIP_PREPARE=${SKIP_PREPARE:-1}"
    "DSV4_HC_MULT=${DSV4_HC_MULT:-4}"
    "IMAGE=${IMAGE:-lumen/dsv4-lumen:mi308x}"
    "V4_SPARSE_MLA_BACKEND=${V4_SPARSE_MLA_BACKEND:-triton}"
    "V4_INDEXER_IMPL=${V4_INDEXER_IMPL:-aiter}"
    "OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
    "DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-180}"
    "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"
    "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}"
    "NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
    "NCCL_IB_GDR_LEVEL=${NCCL_IB_GDR_LEVEL:-}"
    "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-}"
    "NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}"
    "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
    "MEGATRON_NO_BATCH_P2P_COMM=${MEGATRON_NO_BATCH_P2P_COMM:-1}"
    "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
    "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens14np0}"
    "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens14np0}"
)

if [[ "${DSV4_MODE}" == "finetune" ]]; then
    COMMON_ENV+=(
        "DATA_DIR=${DATA_DIR:-${DATA_ROOT}/datasets}"
        "GBS=${GBS:-256}"
        "DSV4_KEEP_GBS=${DSV4_KEEP_GBS:-0}"
        "SEQ_LEN=${SEQ_LEN:-4096}"
        "DSV4_KEEP_SEQ_LEN=${DSV4_KEEP_SEQ_LEN:-0}"
        "ROLLOUT_N_PROMPTS=${ROLLOUT_N_PROMPTS:-32}"
        "ROLLOUT_N_PER_PROMPT=${ROLLOUT_N_PER_PROMPT:-8}"
        "SMOKE_LEGACY_FAKE_ROLLOUT=${SMOKE_LEGACY_FAKE_ROLLOUT:-0}"
        "NUM_ROLLOUT=${NUM_ROLLOUT:-10}"
        "DEBUG_TRAIN_ONLY=${DEBUG_TRAIN_ONLY:-1}"
        "V4_INDEXER_BLOCK_N=${V4_INDEXER_BLOCK_N:-64}"
        "V4_INDEXER_NUM_STAGES=${V4_INDEXER_NUM_STAGES:-1}"
        "TP=${TP:-}"
        "PP=${PP:-}"
        "EP=${EP:-}"
        "ETP=${ETP:-}"
        "DECODER_FIRST_PP_LAYERS=${DECODER_FIRST_PP_LAYERS:-}"
        "DECODER_LAST_PP_LAYERS=${DECODER_LAST_PP_LAYERS:-}"
    )
else
    COMMON_ENV+=(
        "LOAD_CKPT=${LOAD_CKPT:-0}"
        "GBS=${GBS:-8}"
        "SEQ_LEN=${SEQ_LEN:-2048}"
        "TRAIN_ITERS=${TRAIN_ITERS:-10}"
        "EVAL_ITERS=${EVAL_ITERS:-1}"
        "RERUN_MODE=${RERUN_MODE:-disabled}"
        "DSV4_ENABLE_EXPERT_BIAS=${DSV4_ENABLE_EXPERT_BIAS:-0}"
    )
fi

echo "════════════════════════════════════════════════"
echo "  2-node ${DSV4_MODE} launch  PREFLIGHT_ID=${PREFLIGHT_ID}"
echo "  Head   : ${MASTER_ADDR} (NODE_RANK=0)"
echo "  Worker : ${WORKER_SSH} (NODE_RANK=1)"
echo "  MODEL_DIR (head)  : ${MODEL_DIR}"
echo "  MODEL_DIR (worker): ${WORKER_MODEL_DIR}"
if [[ "${DSV4_MODE}" == "finetune" ]]; then
    echo "  GBS=${GBS:-256}  SEQ_LEN=${SEQ_LEN:-4096}  NUM_ROLLOUT=${NUM_ROLLOUT:-10}  DSV4_HC_MULT=${DSV4_HC_MULT:-4}"
    echo "  parallel : TP=${TP:-4} PP=${PP:-4} EP=${EP:-4} (override via env)"
else
    echo "  LOAD_CKPT=${LOAD_CKPT:-0}  GBS=${GBS:-8}  TRAIN_ITERS=${TRAIN_ITERS:-10}  EVAL_ITERS=${EVAL_ITERS:-1}"
fi
echo "  MLA=${V4_SPARSE_MLA_BACKEND:-triton}  OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
echo "════════════════════════════════════════════════"

docker rm -f "${CONTAINER_PREFIX}0" "${CONTAINER_PREFIX}1" 2>/dev/null || true
_ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
if [[ -n "${SSH_KEY}" ]]; then
    _ssh_opts+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes)
fi
if ssh "${_ssh_opts[@]}" \
    "${WORKER_SSH}" "docker rm -f ${CONTAINER_PREFIX}0 ${CONTAINER_PREFIX}1 2>/dev/null || true"; then
    WORKER_REACHABLE=1
else
    WORKER_REACHABLE=0
    echo "[launch][WARN] worker unreachable (${WORKER_SSH}) - head will start; preflight waits for worker manifest"
fi

join_env() {
    local out=""
    local kv=""
    for kv in "${COMMON_ENV[@]}"; do
        out+="${kv} "
    done
    printf '%s' "${out}"
}

join_worker_env() {
    local out=""
    local kv=""
    for kv in "${COMMON_ENV[@]}"; do
        if [[ "${kv}" == MODEL_DIR=* ]]; then
            out+="MODEL_DIR=${WORKER_MODEL_DIR} "
        else
            out+="${kv} "
        fi
    done
    printf '%s' "${out}"
}

HEAD_LOG="${LOG_DIR}/lumen_dsv4_flash_${LOG_TAG}_launch_head_${PREFLIGHT_ID}.log"
WORKER_LOG="${LOG_DIR}/lumen_dsv4_flash_${LOG_TAG}_launch_worker_${PREFLIGHT_ID}.log"

cd "${LUMEN_DIR}"
nohup env NODE_RANK=0 DSV4_MODE="${DSV4_MODE}" DSV4_PROFILE="${DSV4_PROFILE}" $(join_env) \
    bash examples/dsv4/dsv4_launch.sh \
    > "${HEAD_LOG}" 2>&1 &
HEAD_PID=$!
echo "[launch] head pid=${HEAD_PID} log=${HEAD_LOG}"

sleep 5

if [[ "${WORKER_REACHABLE}" == "1" ]]; then
    ssh "${_ssh_opts[@]}" "${WORKER_SSH}" \
        "cd ${LUMEN_DIR} && nohup env NODE_RANK=1 DSV4_MODE=${DSV4_MODE} DSV4_PROFILE=${DSV4_PROFILE} $(join_worker_env) \
            bash examples/dsv4/dsv4_launch.sh \
            > ${WORKER_LOG} 2>&1 & echo worker_pid=\$! log=${WORKER_LOG}"
else
    echo "[launch][WARN] skip worker launch - bring up worker then run:"
    echo "  cd ${LUMEN_DIR} && NODE_RANK=1 DSV4_MODE=${DSV4_MODE} DSV4_PROFILE=${DSV4_PROFILE} $(join_worker_env) bash examples/dsv4/dsv4_launch.sh"
fi

echo "[launch] done - tail training logs:"
echo "  head:   tail -f ${LOG_DIR}/${TRAIN_LOG_GLOB}0_*.log"
echo "  worker: tail -f ${LOG_DIR}/${TRAIN_LOG_GLOB}1_*.log"
