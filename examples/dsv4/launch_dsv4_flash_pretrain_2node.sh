#!/usr/bin/env bash
# Launch DSV4 Flash 2-node pretrain from the head node with identical env on both ranks.
#
# Usage (on head node):
#   cd ~/Lumen
#   MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-host> \
#     MODEL_DIR=/data1/${USER}/models \
#     WORKER_MODEL_DIR=/mnt/nvme0n1/${USER}/models \
#     bash examples/dsv4/launch_dsv4_flash_pretrain_2node.sh
#
# Tier-2 loss correctness probe (LR=0, 2 iters, LOAD_CKPT=1):
#   LOSS_PROBE=1 SKIP_PREPARE=1 bash examples/dsv4/launch_dsv4_flash_pretrain_2node.sh
#
# Optional overrides (exported to both nodes unless noted):
#   LOAD_CKPT=1 GBS=8 TRAIN_ITERS=100 SKIP_PREPARE=1 IMAGE=lumen/dsv4-lumen:mi308x
#   WORKER_MODEL_DIR=/mnt/nvme0n1/${USER}/models  (worker-only host ckpt path)
#   Health (before launch): DSV4_LAUNCH_MAX_MEM_USED_GIB=200 DSV4_LAUNCH_MAX_GPU_USE_PCT=5
#   SKIP_HEALTH_CHECK=1  DSV4_LAUNCH_KILL_STALE_PROCS=0  SKIP_WORKER_RSYNC=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"
# shellcheck source=examples/dsv4/dsv4_launch_common.sh
source "${SCRIPT_DIR}/dsv4_launch_common.sh"

MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to head node IP}"
WORKER_ADDR="${WORKER_ADDR:-}"
WORKER_SSH="${WORKER_SSH:-}"
if [[ -z "${WORKER_SSH}" && -n "${WORKER_ADDR}" ]]; then
    WORKER_SSH="${USER}@${WORKER_ADDR}"
fi
WORKER_SSH="${WORKER_SSH:?Set WORKER_SSH (e.g. ${USER}@worker-host) or WORKER_ADDR}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519_conductor}"
PREFLIGHT_ID="$(date +%Y%m%d_%H%M%S)"
WORKER_MODEL_DIR="${WORKER_MODEL_DIR:-${MODEL_DIR}}"

dsv4_launch_validate_local_model_dirs
dsv4_launch_init_ssh

if [[ "${LOSS_PROBE:-0}" == "1" ]]; then
    TRAIN_ITERS="${LOSS_PROBE_ITERS:-2}"
    EVAL_ITERS=0
    LOAD_CKPT="${LOAD_CKPT:-1}"
fi

COMMON_ENV=(
    "PREFLIGHT_ID=${PREFLIGHT_ID}"
    "MASTER_ADDR=${MASTER_ADDR}"
    "MASTER_PORT=${MASTER_PORT:-29500}"
    "MODEL_DIR=${MODEL_DIR}"
    "DATA_ROOT=${DATA_ROOT}"
    "LOG_DIR=${LOG_DIR}"
    "SKIP_PREPARE=${SKIP_PREPARE:-1}"
    "LOSS_PROBE=${LOSS_PROBE:-0}"
    "LOSS_PROBE_ITERS=${LOSS_PROBE_ITERS:-2}"
    "LOAD_CKPT=${LOAD_CKPT:-1}"
    "GBS=${GBS:-8}"
    "TRAIN_ITERS=${TRAIN_ITERS:-10}"
    "EVAL_ITERS=${EVAL_ITERS:-0}"
    "DSV4_HC_MULT=${DSV4_HC_MULT:-4}"
    "IMAGE=${IMAGE:-lumen/dsv4-lumen:mi308x}"
    "V4_SPARSE_MLA_BACKEND=${V4_SPARSE_MLA_BACKEND:-triton}"
    "MHC_BACKEND=${MHC_BACKEND:-triton}"
    "V4_INDEXER_IMPL=${V4_INDEXER_IMPL:-aiter}"
    "V4_INDEXER_BLOCK_N=${V4_INDEXER_BLOCK_N:-64}"
    "V4_INDEXER_NUM_STAGES=${V4_INDEXER_NUM_STAGES:-1}"
    "TILEKERNELS_DIR=${TILEKERNELS_DIR}"
    "OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
    "DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-360}"
    "NPROC_PER_NODE=${NPROC_PER_NODE:-8}"
    "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    "TP=${TP:-4}"
    "PP=${PP:-4}"
    "EP=${EP:-4}"
    "ETP=${ETP:-1}"
    "DECODER_FIRST_PP_LAYERS=${DECODER_FIRST_PP_LAYERS:-11}"
    "DECODER_LAST_PP_LAYERS=${DECODER_LAST_PP_LAYERS:-10}"
    "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"
    "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}"
    "NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
    "NCCL_IB_GDR_LEVEL=${NCCL_IB_GDR_LEVEL:-0}"
    "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-LOC}"
    "NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}"
    "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
    "MEGATRON_NO_BATCH_P2P_COMM=${MEGATRON_NO_BATCH_P2P_COMM:-1}"
    "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
    "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens14np0}"
    "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens14np0}"
    "MEGATRON_PATH=${MEGATRON_PATH}"
)

echo "════════════════════════════════════════════════"
echo "  2-node pretrain launch  PREFLIGHT_ID=${PREFLIGHT_ID}"
echo "  Head   : ${MASTER_ADDR} (NODE_RANK=0)"
echo "  Worker : ${WORKER_SSH} (NODE_RANK=1)"
echo "  MODEL_DIR (head)  : ${MODEL_DIR}"
echo "  MODEL_DIR (worker): ${WORKER_MODEL_DIR}"
echo "  LOSS_PROBE=${LOSS_PROBE:-0}  LOAD_CKPT=${LOAD_CKPT:-1}  GBS=${GBS:-8}  TRAIN_ITERS=${TRAIN_ITERS:-10}"
echo "  GPUs/node : ${NPROC_PER_NODE:-8}  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
echo "  Parallel  : TP=${TP:-4} PP=${PP:-4} EP=${EP:-4} (${DECODER_FIRST_PP_LAYERS:-11}+${DECODER_LAST_PP_LAYERS:-10})"
echo "  MLA=${V4_SPARSE_MLA_BACKEND:-triton}  indexer=${V4_INDEXER_IMPL:-aiter}  OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
echo "════════════════════════════════════════════════"

dsv4_launch_cleanup_stale lumen-dsv4-full-node0 lumen-dsv4-full-node1
if ssh "${_ssh_opts[@]}" \
    "${WORKER_SSH}" 'docker rm -f lumen-dsv4-full-node0 lumen-dsv4-full-node1 2>/dev/null || true'; then
    WORKER_REACHABLE=1
else
    WORKER_REACHABLE=0
    echo "[launch][WARN] worker unreachable (${WORKER_SSH}) — head will start; preflight waits for worker manifest"
fi

dsv4_launch_run_health_checks "$([[ "${WORKER_REACHABLE}" == "1" ]] && echo "${WORKER_SSH}" || echo "")"
dsv4_launch_publish_preflight_id

if [[ "${WORKER_REACHABLE}" == "1" ]]; then
    dsv4_launch_sync_worker_dsv4_scripts "${WORKER_SSH}"
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

HEAD_LOG="${LOG_DIR}/lumen_dsv4_flash_launch_head_${PREFLIGHT_ID}.log"
WORKER_LOG="${LOG_DIR}/lumen_dsv4_flash_launch_worker_${PREFLIGHT_ID}.log"

cd "${LUMEN_DIR}"
nohup env NODE_RANK=0 $(join_env) \
    bash examples/dsv4/run_dsv4_flash_pretrain.sh \
    > "${HEAD_LOG}" 2>&1 &
HEAD_PID=$!
echo "[launch] head pid=${HEAD_PID} log=${HEAD_LOG}"

sleep 5

if [[ "${WORKER_REACHABLE}" == "1" ]]; then
    ssh "${_ssh_opts[@]}" "${WORKER_SSH}" \
        "cd ${LUMEN_DIR} && nohup env NODE_RANK=1 $(join_worker_env) \
            bash examples/dsv4/run_dsv4_flash_pretrain.sh \
            > ${WORKER_LOG} 2>&1 & echo worker_pid=\$! log=${WORKER_LOG}"
else
    echo "[launch][WARN] skip worker launch — bring up worker then run:"
    echo "  cd ${LUMEN_DIR} && NODE_RANK=1 $(join_worker_env) bash examples/dsv4/run_dsv4_flash_pretrain.sh"
fi

echo "[launch] done — tail training logs:"
echo "  head:   tail -f ${LOG_DIR}/lumen_dsv4_flash_pretrain_node0_*.log"
echo "  worker: tail -f ${LOG_DIR}/lumen_dsv4_flash_pretrain_node1_*.log"
