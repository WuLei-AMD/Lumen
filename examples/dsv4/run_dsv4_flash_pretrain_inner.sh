#!/usr/bin/env bash
# Inner container entry for run_dsv4_flash_pretrain.sh (2-node Flash pretrain smoke).
set -euo pipefail

cd /workspace/Lumen
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"

TRAIN_ITERS="${TRAIN_ITERS:-10}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
LOAD_CKPT="${LOAD_CKPT:-0}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR required}"
MASTER_PORT="${MASTER_PORT:-29500}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-180}"

# shellcheck source=examples/dsv4/dsv4_flash_megatron_args.sh
source examples/dsv4/dsv4_flash_megatron_args.sh
# shellcheck source=examples/dsv4/dsv4_flash_mi300x_parallel.sh
source examples/dsv4/dsv4_flash_mi300x_parallel.sh

if [[ "${LOAD_CKPT}" != "1" ]]; then
    DSV4_MODEL_ARGS+=(--dsv4-n-hash-layers 0)
fi

export LUMEN_DSV4_PRETRAIN=1
export DSV4_ALIGN_RL_ROUTING="${DSV4_ALIGN_RL_ROUTING:-1}"
export MILES_DSV4_AITER_HASH_ROUTING="${MILES_DSV4_AITER_HASH_ROUTING:-1}"
export MILES_DSV4_HASH_ROUTING_CPU="${MILES_DSV4_HASH_ROUTING_CPU:-1}"
export MILES_DSV4_USE_TORCH_OPTIMIZER="${MILES_DSV4_USE_TORCH_OPTIMIZER:-1}"
# MI308X: RL Megatron fork may route MHC through miles_plugins (TileLang default).
# TileKernels MHC needs 98304 B smem > gfx942 65536 B limit; force AIter like Miles pretrain.
export MILES_DSV4_MHC_BACKEND="${MILES_DSV4_MHC_BACKEND:-aiter}"
# shellcheck source=examples/dsv4/setup_container_env.sh
source examples/dsv4/setup_container_env.sh
setup_dsv4_container_env

CKPT="${DSV4_CKPT_PATH:-/root/models/${MODEL_NAME}_torch_dist}"
if [[ ! -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
    FALLBACK="/root/models/${MODEL_NAME}_torch_dist_hc${DSV4_HC_MULT}"
    if [[ -f "${FALLBACK}/latest_checkpointed_iteration.txt" ]]; then
        echo "[prepare] using fallback checkpoint ${FALLBACK}"
        CKPT="${FALLBACK}"
    fi
    ALT="/root/models/DeepSeek-V4-Flash-FP8_torch_dist_Lumen"
    if [[ ! -f "${CKPT}/latest_checkpointed_iteration.txt" && -f "${ALT}/latest_checkpointed_iteration.txt" ]]; then
        echo "[prepare] using Lumen torch_dist ${ALT}"
        CKPT="${ALT}"
    fi
fi

if [[ "${LOAD_CKPT}" == "1" ]]; then
    export DSV4_ROLLOUT_TOKENS_PATH="${DSV4_ROLLOUT_TOKENS_PATH:-/root/models/dsv4_rollout_tokens.pt}"
    if [[ ! -f "${DSV4_ROLLOUT_TOKENS_PATH}" && "${DSV4_AUTO_ROLLOUT_TOKENS:-1}" == "1" && -d /workspace/miles ]]; then
        echo "[prepare] generating rollout tokens for hash routing from ${CKPT} ..."
        python3 /workspace/miles/scripts/dsv4/generate_dsv4_rollout_tokens.py \
            --output "${DSV4_ROLLOUT_TOKENS_PATH}" \
            --torch-dist-checkpoint "${CKPT}" \
            --num-sequences "${DSV4_ROLLOUT_NUM_SEQUENCES:-256}" \
            --seq-len "${SEQ_LEN:-2048}" \
            --hash-layer "${DSV4_ROLLOUT_HASH_LAYER:-0}" || true
    fi
    if [[ -f "${DSV4_ROLLOUT_TOKENS_PATH}" ]]; then
        echo "[pretrain-full] using rollout tokens ${DSV4_ROLLOUT_TOKENS_PATH}"
    else
        echo "[pretrain-full][WARN] rollout tokens missing — mock data may hit unmapped tid2eid slots"
    fi
fi

if [[ "${SKIP_PREPARE}" != "1" && ! -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
    if [[ ! -d /workspace/miles ]]; then
        echo "[ERROR] Checkpoint missing and MILES_DIR not mounted for prepare_dsv4_flash_checkpoint.py"
        exit 1
    fi
    export PYTHONPATH="/workspace/Lumen:/workspace/miles:${PYTHONPATH:-}"
    python examples/dsv4/prepare_dsv4_flash_checkpoint.py
else
    echo "[prepare] torch_dist checkpoint already present — skipping (path=${CKPT})"
fi

LOAD_ARGS=()
if [[ "${LOAD_CKPT}" == "1" && -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
    LOAD_ARGS=(--load "${CKPT}" --no-load-optim --no-load-rng)
    echo "[pretrain-full] loading checkpoint ${CKPT}"
else
    echo "[pretrain-full] training from random init (LOAD_CKPT=${LOAD_CKPT})"
fi

RECOMPUTE_ARGS=()
if [[ "${DSV4_ENABLE_RECOMPUTE:-1}" == "1" ]]; then
    RECOMPUTE_ARGS=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
    )
fi

# shellcheck source=examples/dsv4/dsv4_pretrain_repro.sh
source examples/dsv4/dsv4_pretrain_repro.sh
dsv4_pretrain_setup_repro
dsv4_pretrain_print_repro

echo "[pretrain-full] launching torchrun ${NNODES}×${NPROC_PER_NODE} (node_rank=${NODE_RANK}) ..."
echo "[pretrain-full] parallel: TP=${TP} PP=${PP} EP=${EP} | batch GBS=${GBS} MBS=${MBS} seq=${SEQ_LEN}"
echo "[pretrain-full] optimizer CPU offload fraction=${OPTIMIZER_OFFLOAD_FRACTION}"
echo "[pretrain-full] CPU memory: num_workers=0, pin_cpu_grads/params=off (mock-data smoke)"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    examples/dsv4/pretrain_dsv4_megatron.py \
    "${DSV4_MODEL_ARGS[@]}" \
    --transformer-impl local \
    --disable-jit-fuser \
    --no-persist-layer-norm \
    --moe-router-freeze-gate \
    --freeze-e-score-correction-bias \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --decoder-first-pipeline-num-layers "${DECODER_FIRST_PP_LAYERS}" \
    --decoder-last-pipeline-num-layers "${DECODER_LAST_PP_LAYERS}" \
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
    --num-workers 0 \
    --no-pin-cpu-grads \
    --no-pin-cpu-params \
    --split 100,0,0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --accumulate-allreduce-grads-in-fp32 \
    --use-distributed-optimizer \
    --optimizer adam \
    --optimizer-cpu-offload \
    --use-precision-aware-optimizer \
    --overlap-cpu-optimizer-d2h-h2d \
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}" \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --save-interval 1000000 \
    --eval-interval 1000000 \
    --eval-iters "${EVAL_ITERS:-1}" \
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}" \
    "${DSV4_PRETRAIN_REPRO_ARGS[@]}" \
    "${LOAD_ARGS[@]}" \
    --distributed-backend nccl

echo ""
echo "=== [done] Lumen DSV4 Flash full-model pretrain smoke completed (node_rank=${NODE_RANK}) ==="
