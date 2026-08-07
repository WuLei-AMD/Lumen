#!/usr/bin/env bash
# Unified inner container entry for DSV4 pretrain/profile (4layer single-node or flash 2-node).
#
# DSV4_PROFILE=4layer|flash (default 4layer)
# DSV4_MODE=pretrain|profile (default pretrain; profile is 4layer-only)
set -euo pipefail

cd /workspace/Lumen
export CUDA_DEVICE_MAX_CONNECTIONS=1

DSV4_PROFILE="${DSV4_PROFILE:-4layer}"
DSV4_MODE="${DSV4_MODE:-pretrain}"

if [[ "${DSV4_MODE}" == "profile" && "${DSV4_PROFILE}" != "4layer" ]]; then
    echo "[${DSV4_MODE}] ERROR: profiler only supports DSV4_PROFILE=4layer"
    exit 1
fi

if [[ "${DSV4_PROFILE}" == "flash" ]]; then
    export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
    MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8}"
    NNODES="${NNODES:-2}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    NODE_RANK="${NODE_RANK:-0}"
    MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR required}"
    MASTER_PORT="${MASTER_PORT:-29500}"
    OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
    DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-180}"
    # Megatron default validate_results reruns each iter 2-3x; disable for flash
    # smoke unless explicitly enabled (43L iter-3 crash bisect).
    RERUN_MODE="${RERUN_MODE:-disabled}"
else
    MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8-4layer}"
    NNODES=1
    NPROC_PER_NODE=8
    NODE_RANK=0
fi

if [[ "${DSV4_MODE}" == "profile" ]]; then
    export TMPDIR="${TMPDIR:-/dev/shm/lumen-dsv4-tmp}"
    mkdir -p "${TMPDIR}"
    TRAIN_ITERS="${TRAIN_ITERS:-5}"
    SKIP_PREPARE="${SKIP_PREPARE:-1}"
    LOAD_CKPT="${LOAD_CKPT:-1}"
    EVAL_ITERS="${EVAL_ITERS:-0}"
    RESULTS_DIR="${RESULTS_DIR:-/workspace/Lumen/examples/dsv4/results}"
    mkdir -p "${RESULTS_DIR}"
    LUMEN_PROF_START="${LUMEN_PROF_START:-3}"
    LUMEN_PROF_END="${LUMEN_PROF_END:-5}"
    LUMEN_PROF_STOP_AFTER="${LUMEN_PROF_STOP_AFTER:-${LUMEN_PROF_END}}"
    export LUMEN_PROF_START LUMEN_PROF_END LUMEN_PROF_STOP_AFTER
    export LUMEN_PROF_OUTPUT="${LUMEN_PROF_OUTPUT:-${RESULTS_DIR}/dsv4_4layer_profile.txt}"
    export LUMEN_PROF_XLSX="${LUMEN_PROF_XLSX:-${RESULTS_DIR}/dsv4_4layer_operator_breakdown.xlsx}"
    export LUMEN_PROF_TRACE="${LUMEN_PROF_TRACE:-}"
    export LUMEN_PROF_SHAPES="${LUMEN_PROF_SHAPES:-0}"
else
    TRAIN_ITERS="${TRAIN_ITERS:-10}"
    SKIP_PREPARE="${SKIP_PREPARE:-0}"
    LOAD_CKPT="${LOAD_CKPT:-0}"
    EVAL_ITERS="${EVAL_ITERS:-1}"
fi
export TRAIN_ITERS

# shellcheck source=examples/dsv4/dsv4_megatron_args.sh
source examples/dsv4/dsv4_megatron_args.sh
# shellcheck source=examples/dsv4/dsv4_pretrain_common.sh
source examples/dsv4/dsv4_pretrain_common.sh

export LUMEN_DSV4_PRETRAIN=1
# shellcheck source=examples/dsv4/setup_container_env.sh
source examples/dsv4/setup_container_env.sh
setup_dsv4_container_env

dsv4_resolve_pretrain_ckpt "${MODEL_NAME}" "${DSV4_PROFILE}"
dsv4_prepare_pretrain_ckpt "${DSV4_PROFILE}" "${SKIP_PREPARE}"
dsv4_pretrain_load_args
dsv4_pretrain_recompute_args

TAG="${DSV4_MODE}"
if [[ "${DSV4_PROFILE}" == "flash" ]]; then
    TAG="${DSV4_MODE}-full"
fi

TORCHRUN_EXTRA=()
if [[ "${DSV4_PROFILE}" == "flash" ]]; then
    echo "[${TAG}] launching torchrun ${NNODES}×${NPROC_PER_NODE} (node_rank=${NODE_RANK}) ..."
    echo "[${TAG}] parallel: TP=${TP} PP=${PP} EP=${EP} | batch GBS=${GBS} MBS=${MBS} seq=${SEQ_LEN}"
    echo "[${TAG}] optimizer CPU offload fraction=${OPTIMIZER_OFFLOAD_FRACTION}"
    TORCHRUN=(
        torchrun
        --nnodes="${NNODES}"
        --nproc_per_node="${NPROC_PER_NODE}"
        --node_rank="${NODE_RANK}"
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
    )
    TORCHRUN_EXTRA=(
        --decoder-first-pipeline-num-layers "${DECODER_FIRST_PP_LAYERS}"
        --decoder-last-pipeline-num-layers "${DECODER_LAST_PP_LAYERS}"
        --num-workers 0
        --no-pin-cpu-grads
        --no-pin-cpu-params
        --optimizer-cpu-offload
        --use-precision-aware-optimizer
        --overlap-cpu-optimizer-d2h-h2d
        --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
        --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}"
    )
else
    echo "[${TAG}] launching torchrun (native Megatron, no Ray) ..."
    echo "[${TAG}] batch: GBS=${GBS} MBS=${MBS} seq_len=${SEQ_LEN} (hc_mult=${DSV4_HC_MULT})"
    if [[ "${DSV4_MODE}" == "profile" ]]; then
        echo "[profile] torch.profiler steps ${LUMEN_PROF_START}-${LUMEN_PROF_END}, stop after ${LUMEN_PROF_STOP_AFTER}"
        echo "[profile] outputs: ${LUMEN_PROF_OUTPUT}"
        echo "[profile]           ${LUMEN_PROF_XLSX}"
    fi
    TORCHRUN=(torchrun --nproc_per_node="${NPROC_PER_NODE}" --nnodes=1)
fi

"${TORCHRUN[@]}" \
    examples/dsv4/pretrain_dsv4_megatron.py \
    "${DSV4_MODEL_ARGS[@]}" \
    --transformer-impl local \
    --disable-jit-fuser \
    --moe-router-freeze-gate \
    --freeze-e-score-correction-bias \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --context-parallel-size "${CP}" \
    --expert-model-parallel-size "${EP}" \
    --expert-tensor-parallel-size "${ETP}" \
    --sequence-parallel \
    "${RECOMPUTE_ARGS[@]}" \
    --micro-batch-size "${MBS}" \
    --global-batch-size "${GBS}" \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings "${SEQ_LEN}" \
    --train-iters "${TRAIN_ITERS}" \
    --mock-data \
    --split 100,0,0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --accumulate-allreduce-grads-in-fp32 \
    --use-distributed-optimizer \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --save-interval 1000000 \
    --eval-interval 1000000 \
    --eval-iters "${EVAL_ITERS}" \
    --rerun-mode "${RERUN_MODE:-disabled}" \
    "${TORCHRUN_EXTRA[@]}" \
    "${LOAD_ARGS[@]}" \
    --distributed-backend nccl

echo ""
if [[ "${DSV4_MODE}" == "profile" ]]; then
    echo "=== [done] DSV4 4-layer profiler completed ==="
    echo "Profile txt : ${LUMEN_PROF_OUTPUT}"
    echo "Operator xlsx: ${LUMEN_PROF_XLSX}"
else
    echo "=== [done] Lumen DSV4 ${DSV4_PROFILE} Megatron pretrain smoke completed ==="
fi
