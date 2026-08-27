#!/usr/bin/env bash
# Qwen3-30B-A3B, one MI350X node (8 GPUs), EP=8.

NGPU=${NGPU:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-8}
ETP=${ETP:-1}

SEQ_LEN=${SEQ_LEN:-4096}
MBS=${MBS:-1}
GBS=${GBS:-8}
TRAIN_STEPS=${TRAIN_STEPS:-20}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-2}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-0}
LUMEN_ATTN_BACKEND=${LUMEN_ATTN_BACKEND:-triton}

TOKENIZER_PATH=${TOKENIZER_PATH:-/workspace/Lumen/examples/qwen3/tokenizer}
DATA_PATH=${DATA_PATH:-/workspace/Lumen/examples/qwen3-30b-a3b/data/mock_train.jsonl}
RESULTS_DIR=${RESULTS_DIR:-/workspace/Lumen/examples/qwen3-30b-a3b/results}

# sequential | te_grouped | sonic
MOE_IMPL=${MOE_IMPL:-sequential}
