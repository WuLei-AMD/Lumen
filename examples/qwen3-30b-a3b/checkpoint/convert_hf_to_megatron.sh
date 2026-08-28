#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_PATH=${MEGATRON_PATH:-/workspace/Megatron-LM}
HF_DIR=${HF_DIR:-/nobackup/model/Qwen3-30B-A3B}
SAVE_DIR=${SAVE_DIR:-/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8}

mkdir -p "${SAVE_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${MEGATRON_PATH}/tools/checkpoint:${MEGATRON_PATH}:${PYTHONPATH:-}"

python3 "${MEGATRON_PATH}/tools/checkpoint/convert.py" \
    --model-type GPT \
    --loader qwen3_moe_hf \
    --saver qwen3_moe \
    --load-dir "${HF_DIR}" \
    --save-dir "${SAVE_DIR}" \
    --megatron-path "${MEGATRON_PATH}" \
    --tokenizer-model "${HF_DIR}" \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 8 \
    --saver-transformer-impl transformer_engine \
    --max-queue-size 2
