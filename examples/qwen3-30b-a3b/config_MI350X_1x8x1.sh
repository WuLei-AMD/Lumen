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
# Perf defaults for the Megatron SonicMoE path (override to A/B):
#   LUMEN_ATTN_BACKEND=csrc          AITER CK fmha_v3
#   OVERLAP_MOE_EP_COMM=1            combined 1F1B EP all-to-all overlap
#   CUDA_DEVICE_MAX_CONNECTIONS=8    (set in run_docker.sh)
#   CUDA_GRAPH_SCOPE=attn            Megatron TE graphs on _forward_attention
#   CUDA_GRAPH_SCOPE=none            disables graphs
#   SONIC grouped GEMM: triton + Qwen3 tuned configs (run_docker.sh)
LUMEN_ATTN_BACKEND=${LUMEN_ATTN_BACKEND:-csrc}
OVERLAP_MOE_EP_COMM=${OVERLAP_MOE_EP_COMM:-1}
CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL:-transformer_engine}
CUDA_GRAPH_SCOPE=${CUDA_GRAPH_SCOPE:-attn}

TOKENIZER_PATH=${TOKENIZER_PATH:-/workspace/Lumen/examples/qwen3/tokenizer}
DATA_PATH=${DATA_PATH:-/workspace/Lumen/examples/qwen3-30b-a3b/data/mock_train.jsonl}
RESULTS_DIR=${RESULTS_DIR:-/workspace/Lumen/examples/qwen3-30b-a3b/results}

# sequential | te_grouped | sonic
MOE_IMPL=${MOE_IMPL:-sequential}
