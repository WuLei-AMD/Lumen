#!/usr/bin/env bash
# Launch DSV4 Flash 2-node pretrain from the head node with identical env on both ranks.
#
# Usage (on head node):
#   cd ~/Lumen
#   MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-host> \
#     bash examples/dsv4/launch_dsv4_flash_pretrain_2node.sh
#
# Optional overrides (exported to both nodes):
#   LOAD_CKPT=0 GBS=8 TRAIN_ITERS=10 SKIP_PREPARE=1 IMAGE=lumen/dsv4-lumen:mi308x

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"

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
WORKER_LOG_DIR="${WORKER_LOG_DIR:-${LOG_DIR}}"
WORKER_DATA_ROOT="${WORKER_DATA_ROOT:-${DATA_ROOT}}"
WORKER_NCCL_IF="${WORKER_NCCL_IF:-${NCCL_SOCKET_IFNAME:-enp193s0f0}}"

if [[ "${DSV4_DUMP_LAYER_ACT:-0}" == "1" ]]; then
    DSV4_DUMP_STOP_AFTER_LAYER="${DSV4_DUMP_STOP_AFTER_LAYER:-1}"
    EVAL_ITERS="${EVAL_ITERS:-0}"
    TRAIN_ITERS="${TRAIN_ITERS:-1}"
fi

COMMON_ENV=(
    "PREFLIGHT_ID=${PREFLIGHT_ID}"
    "MASTER_ADDR=${MASTER_ADDR}"
    "LUMEN_DIR=${LUMEN_DIR}"
    "AITER_DIR=${AITER_DIR}"
    "MILES_DIR=${MILES_DIR}"
    "MODEL_DIR=${MODEL_DIR}"
    "DATA_ROOT=${DATA_ROOT}"
    "LOG_DIR=${LOG_DIR}"
    "SKIP_PREPARE=${SKIP_PREPARE:-1}"
    "SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-1}"
    "LOAD_CKPT=${LOAD_CKPT:-0}"
    "DSV4_DUMP_LAYER_ACT=${DSV4_DUMP_LAYER_ACT:-0}"
    "DSV4_DUMP_DIR=${DSV4_DUMP_DIR:-/root/models/dsv4_layer_act_lumen}"
    "DSV4_DUMP_TAG=${DSV4_DUMP_TAG:-lumen}"
    "DSV4_DUMP_LAYER_ACT_MAX=${DSV4_DUMP_LAYER_ACT_MAX:-1}"
    "DSV4_DUMP_LAYER1_INNER=${DSV4_DUMP_LAYER1_INNER:-1}"
    "DSV4_DUMP_STOP_AFTER_LAYER=${DSV4_DUMP_STOP_AFTER_LAYER:-}"
    "DSV4_DUMP_GRAD_NORM=${DSV4_DUMP_GRAD_NORM:-0}"
    "DSV4_DUMP_GRAD_NORM_DIR=${DSV4_DUMP_GRAD_NORM_DIR:-/root/models/dsv4_gn_dump_lumen}"
    "DSV4_DUMP_ATTN_IO=${DSV4_DUMP_ATTN_IO:-0}"
    "DSV4_DUMP_ATTN_IO_DIR=${DSV4_DUMP_ATTN_IO_DIR:-/root/models/dsv4_attn_io_lumen}"
    "DSV4_DUMP_MHC_DX=${DSV4_DUMP_MHC_DX:-0}"
    "DSV4_DUMP_MHC_DX_DIR=${DSV4_DUMP_MHC_DX_DIR:-/root/models/dsv4_mhc_dx_lumen}"
    "GBS=${GBS:-8}"
    "SEQ_LEN=${SEQ_LEN:-2048}"
    "TRAIN_ITERS=${TRAIN_ITERS:-10}"
    "PRETRAIN_LR=${PRETRAIN_LR:-1e-6}"
    "PRETRAIN_MIN_LR=${PRETRAIN_MIN_LR:-${PRETRAIN_LR:-1e-6}}"
    "EVAL_ITERS=${EVAL_ITERS:-1}"
    "DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-360}"
    "DSV4_HC_MULT=${DSV4_HC_MULT:-4}"
    "IMAGE=${IMAGE:-lumen/dsv4-lumen:mi308x}"
    "V4_SPARSE_MLA_BACKEND=${V4_SPARSE_MLA_BACKEND:-triton}"
    "DSV4_FORCE_MLA_BACKEND=${DSV4_FORCE_MLA_BACKEND:-}"
    "V4_INDEXER_IMPL=${V4_INDEXER_IMPL:-aiter}"
    "LUMEN_DSV4_LOCAL_RMSNORM=${LUMEN_DSV4_LOCAL_RMSNORM:-0}"
    "LUMEN_DSV4_DUP_TORCH_LINEAR=${LUMEN_DSV4_DUP_TORCH_LINEAR:-0}"
    "OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
    "EXP_AVG_DTYPE=${EXP_AVG_DTYPE:-bf16}"
    "EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE:-bf16}"
    "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp193s0f0}"
    "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp193s0f0}"
    "NCCL_IB_GDR_LEVEL=${NCCL_IB_GDR_LEVEL:-0}"
    "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-LOC}"
    "MEGATRON_NO_BATCH_P2P_COMM=${MEGATRON_NO_BATCH_P2P_COMM:-1}"
    "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
)

echo "════════════════════════════════════════════════"
echo "  2-node launch  PREFLIGHT_ID=${PREFLIGHT_ID}"
echo "  Head   : ${MASTER_ADDR} (NODE_RANK=0)"
echo "  Worker : ${WORKER_SSH} (NODE_RANK=1)"
echo "  Head MODEL_DIR=${MODEL_DIR}  Worker MODEL_DIR=${WORKER_MODEL_DIR}"
echo "  LOAD_CKPT=${LOAD_CKPT:-0}  GBS=${GBS:-8}  SEQ_LEN=${SEQ_LEN:-2048}  TRAIN_ITERS=${TRAIN_ITERS:-10}"
echo "  IMAGE=${IMAGE:-lumen/dsv4-lumen:mi308x}  AITER_DIR=${AITER_DIR}"
echo "  MLA=${V4_SPARSE_MLA_BACKEND:-triton}  OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
echo "════════════════════════════════════════════════"

_ssh_opts=(-i "${SSH_KEY}" -o StrictHostKeyChecking=no -o BatchMode=yes -o IdentitiesOnly=yes)
_rsync_rsh="ssh ${_ssh_opts[*]}"
ssh "${_ssh_opts[@]}" "${WORKER_SSH}" \
    "mkdir -p ${LUMEN_DIR}/lumen/models/dsv4/megatron ${LUMEN_DIR}/lumen/models/dsv4/ops ${LUMEN_DIR}/lumen/modules ${LUMEN_DIR}/lumen/ops/normalization ${LUMEN_DIR}/lumen/models ${LUMEN_DIR}/examples/dsv4/tools" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/modules/parallel_linear.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/modules/parallel_linear.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/models/spec_provider.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/models/spec_provider.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/ops/normalization/rmsnorm.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/ops/normalization/rmsnorm.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/ops/normalization/layernorm.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/ops/normalization/layernorm.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/models/dsv4/ops/hyper_connection.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/models/dsv4/ops/hyper_connection.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/models/dsv4/megatron/deepseek_v4.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/models/dsv4/megatron/deepseek_v4.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/models/dsv4/megatron/layers.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/models/dsv4/megatron/layers.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/models/dsv4/megatron/spec_provider.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/models/dsv4/megatron/spec_provider.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/ops/dsv4/sparse_mla.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/ops/dsv4/sparse_mla.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/models/dsv4/megatron/act_dump.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/models/dsv4/megatron/act_dump.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/lumen/models/dsv4/megatron/pretrain.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/lumen/models/dsv4/megatron/pretrain.py" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/examples/dsv4/run_dsv4_flash_pretrain.sh" \
    "${WORKER_SSH}:${LUMEN_DIR}/examples/dsv4/run_dsv4_flash_pretrain.sh" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/examples/dsv4/run_dsv4_flash_pretrain_inner.sh" \
    "${WORKER_SSH}:${LUMEN_DIR}/examples/dsv4/run_dsv4_flash_pretrain_inner.sh" 2>/dev/null || true
rsync -a -e "${_rsync_rsh}" \
    "${LUMEN_DIR}/examples/dsv4/patch_rocm_megatron_dsv4.py" \
    "${WORKER_SSH}:${LUMEN_DIR}/examples/dsv4/patch_rocm_megatron_dsv4.py" 2>/dev/null || true

docker rm -f lumen-dsv4-full-node0 lumen-dsv4-full-node1 2>/dev/null || true
if ssh -o BatchMode=yes -i "${SSH_KEY}" -o IdentitiesOnly=yes -o ConnectTimeout=10 \
    "${WORKER_SSH}" 'docker rm -f lumen-dsv4-full-node0 lumen-dsv4-full-node1 2>/dev/null || true'; then
    WORKER_REACHABLE=1
else
    WORKER_REACHABLE=0
    echo "[launch][WARN] worker unreachable (${WORKER_SSH}) — head will start; preflight waits for worker manifest"
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
    local out="" kv=""
    for kv in "${COMMON_ENV[@]}"; do
        case "${kv}" in
            MODEL_DIR=*) out+="MODEL_DIR=${WORKER_MODEL_DIR} " ;;
            LOG_DIR=*) out+="LOG_DIR=${WORKER_LOG_DIR} " ;;
            DATA_ROOT=*) out+="DATA_ROOT=${WORKER_DATA_ROOT} " ;;
            NCCL_SOCKET_IFNAME=*) out+="NCCL_SOCKET_IFNAME=${WORKER_NCCL_IF} " ;;
            GLOO_SOCKET_IFNAME=*) out+="GLOO_SOCKET_IFNAME=${WORKER_NCCL_IF} " ;;
            *) out+="${kv} " ;;
        esac
    done
    printf '%s' "${out}"
}

HEAD_LOG="${LOG_DIR}/lumen_dsv4_flash_launch_head_${PREFLIGHT_ID}.log"
WORKER_LOG="${WORKER_LOG_DIR}/lumen_dsv4_flash_launch_worker_${PREFLIGHT_ID}.log"

cd "${LUMEN_DIR}"
nohup env NODE_RANK=0 $(join_env) \
    bash examples/dsv4/run_dsv4_flash_pretrain.sh \
    > "${HEAD_LOG}" 2>&1 &
HEAD_PID=$!
echo "[launch] head pid=${HEAD_PID} log=${HEAD_LOG}"

sleep 5

if [[ "${WORKER_REACHABLE}" == "1" ]]; then
    ssh -o BatchMode=yes -i "${SSH_KEY}" -o IdentitiesOnly=yes "${WORKER_SSH}" \
        "mkdir -p ${WORKER_LOG_DIR} ${WORKER_MODEL_DIR} && cd ${LUMEN_DIR} && nohup env NODE_RANK=1 $(join_worker_env) \
            bash examples/dsv4/run_dsv4_flash_pretrain.sh \
            > ${WORKER_LOG} 2>&1 & echo worker_pid=\$! log=${WORKER_LOG}"
else
    echo "[launch][WARN] skip worker launch — bring up worker then run:"
    echo "  cd ${LUMEN_DIR} && NODE_RANK=1 $(join_worker_env) bash examples/dsv4/run_dsv4_flash_pretrain.sh"
fi

echo "[launch] done — tail training logs:"
echo "  head:   tail -f ${LOG_DIR}/lumen_dsv4_flash_pretrain_node0_*.log"
echo "  worker: tail -f ${LOG_DIR}/lumen_dsv4_flash_pretrain_node1_*.log"
