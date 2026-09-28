#!/bin/bash
# Megatron config for Qwen3-Coder-30B-A3B-Instruct on 1x8 MI308X

# Model
export MODEL_SIZE=30b-a3b
export NUM_LAYERS=48
export HIDDEN_SIZE=2048
export FFN_HIDDEN_SIZE=6144
export NUM_ATTENTION_HEADS=32
export NUM_QUERY_GROUPS=4
export NUM_EXPERTS=128
export MOE_FFN_HIDDEN_SIZE=768
export MOE_ROUTER_TOPK=8
export SEQ_LENGTH=12288
export MAX_POSITION_EMBEDDINGS=262144
export ROTARY_BASE=10000000

# Parallelism: TP=1, PP=1, EP=8, DP=1 (8 GPUs)
export TP=1
export PP=1
export EP=8
export MBS=1
export GBS=8  # global batch size = DP * MBS = 1 * 8

# Training
export LR=0.0001
export MIN_LR=0.00001
export TRAIN_ITERS=1000
export EVAL_INTERVAL=100
export SAVE_INTERVAL=100

# LoRA
export LORA_RANK=32
export LORA_ALPHA=64

# Paths
export HF_MODEL=/home/danyzhan/Lumen/experiments/GEAK-agent-coder/models/Qwen3-Coder-30B-A3B-Instruct
export MEGATRON_CHECKPOINT=/home/danyzhan/Lumen/experiments/GEAK-agent-coder/checkpoints/megatron-qwen3-coder
export DATA_PATH=/home/danyzhan/Lumen/experiments/GEAK-agent-coder/data/build/qwen3_30b_a3b_phase1
export OUTPUT_DIR=/home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/megatron-qwen3-coder

# Precision
export FP8=1
export EXPERT_BACKEND=sonic
