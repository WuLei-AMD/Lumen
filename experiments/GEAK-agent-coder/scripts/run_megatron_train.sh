#!/bin/bash
set -euo pipefail

# Source config
source "$(dirname "$0")/../configs/megatron/config_MI308X_qwen3_coder.sh"

MEGATRON_PATH=/workspace/Megatron-LM
LUMEN_PATH=/home/danyzhan/Lumen
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${LUMEN_PATH}:${LUMEN_PATH}/examples/qwen3-30b-a3b:${MEGATRON_PATH}:${PYTHONPATH:-}"

# Data paths - Megatron needs preprocessed data or we use custom dataset provider
TRAIN_DATA="${DATA_PATH}/train.tokenized.jsonl"
VALID_DATA="${DATA_PATH}/dev.tokenized.jsonl"

torchrun \
    --nnodes=1 \
    --nproc-per-node=8 \
    --standalone \
    ${LUMEN_PATH}/examples/qwen3-30b-a3b/pretrain_qwen3_30b_a3b_megatron.py \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --group-query-attention \
    --num-query-groups ${NUM_QUERY_GROUPS} \
    --num-experts ${NUM_EXPERTS} \
    --moe-ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE} \
    --moe-router-topk ${MOE_ROUTER_TOPK} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --rotary-base ${ROTARY_BASE} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters ${TRAIN_ITERS} \
    --eval-interval ${EVAL_INTERVAL} \
    --save-interval ${SAVE_INTERVAL} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --lr-warmup-iters 10 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --bf16 \
    --no-bias-linear \
    --swiglu \
    --normalization RMSNorm \
    --disable-bias-linear \
    --position-embedding-type rope \
    --no-position-embedding \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --expert-model-parallel-size ${EP} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --load ${MEGATRON_CHECKPOINT} \
    --save ${OUTPUT_DIR} \
    --train-data-path ${TRAIN_DATA} \
    --valid-data-path ${VALID_DATA} \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${HF_MODEL} \
    --expert-backend ${EXPERT_BACKEND} \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 48 \
    --log-interval 1 \
    --size ${MODEL_SIZE} \
    "$@"
