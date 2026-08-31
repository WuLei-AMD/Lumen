#!/usr/bin/env bash
# Qwen3-30B-A3B Megatron pretraining with selectable expert implementation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG=${CONFIG:-"${SCRIPT_DIR}/config_MI350X_1x8x1.sh"}
source "${CONFIG}"
LOAD_ARGS=()
if [ -n "${MEGATRON_LOAD_PATH:-}" ]; then
    LOAD_ARGS=(
        --load "${MEGATRON_LOAD_PATH}"
        --ckpt-format torch
        --finetune
        --no-load-optim
        --no-load-rng
        --exit-on-missing-checkpoint
    )
fi

case "${MOE_IMPL}" in
    sequential)
        MOE_ARGS=""
        ;;
    te_grouped)
        MOE_ARGS="--moe-grouped-gemm"
        ;;
    sonic)
        MOE_ARGS="--lumen-sonic-moe"
        ;;
    *)
        echo "ERROR: MOE_IMPL must be sequential, te_grouped, or sonic" >&2
        exit 2
        ;;
esac

PROFILE_ARGS=()
if [ "${ENABLE_PROFILER:-0}" = "1" ]; then
    PROFILE_DIR=${PROFILE_DIR:-"${RESULTS_DIR}/profile-${MOE_IMPL}"}
    mkdir -p "${PROFILE_DIR}/tensorboard"
    PROFILE_ARGS=(
        --profile
        --use-pytorch-profiler
        --profile-step-start "${PROFILE_STEP_START:-5}"
        --profile-step-end "${PROFILE_STEP_END:-6}"
        --profile-ranks 0
        --tensorboard-dir "${PROFILE_DIR}/tensorboard"
    )
fi

OVERLAP_ARGS=()
if [ "${MEGATRON_OVERLAP:-1}" = "1" ]; then
    OVERLAP_ARGS=(--overlap-grad-reduce --overlap-param-gather)
fi

mkdir -p "$(dirname "${DATA_PATH}")" "${RESULTS_DIR}"
REQUIRED_DOCUMENTS=$((GBS * (TRAIN_STEPS + 4)))
if [ "${REQUIRED_DOCUMENTS}" -lt 128 ]; then
    REQUIRED_DOCUMENTS=128
fi
CURRENT_DOCUMENTS=0
if [ -s "${DATA_PATH}" ]; then
    CURRENT_DOCUMENTS=$(wc -l < "${DATA_PATH}")
fi
if [ "${CURRENT_DOCUMENTS}" -lt "${REQUIRED_DOCUMENTS}" ]; then
    python3 - "${DATA_PATH}" "${SEQ_LEN}" "${GBS}" "${TRAIN_STEPS}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
seq_len, global_batch, steps = map(int, sys.argv[2:])
documents = max(128, global_batch * (steps + 4))
# Use at least one sequence worth of words per document. Tokenizer expansion
# varies, so the previous seq_len // 2 estimate could exhaust long smoke runs.
words = max(1024, seq_len)
text = " ".join(["training"] * words)
with path.open("w") as output:
    for index in range(documents):
        output.write(json.dumps({"text": f"{index} {text}"}) + "\n")
print(f"Generated {documents} mock documents at {path}")
PY
fi

RUN_SUFFIX=${RUN_SUFFIX:-}
RUN_NAME="qwen3-30b-a3b-${MOE_IMPL}${RUN_SUFFIX:+-${RUN_SUFFIX}}-seq${SEQ_LEN}-mbs${MBS}-gbs${GBS}"
LOG_FILE="${RESULTS_DIR}/${RUN_NAME}.log"

echo "Qwen3-30B-A3B: MOE_IMPL=${MOE_IMPL}, TP=${TP}, EP=${EP}, seq=${SEQ_LEN}"

torchrun \
    --nproc_per_node="${NGPU}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/pretrain_qwen3_30b_a3b_megatron.py" \
    --backend megatron \
    --lumen-attn-backend "${LUMEN_ATTN_BACKEND}" \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 4 \
    --kv-channels 128 \
    --make-vocab-size-divisible-by 1187 \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings 4096 \
    --use-rotary-position-embeddings \
    --rotary-base 1000000 \
    --no-position-embedding \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --qk-layernorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-bias-swiglu-fusion \
    --num-experts 128 \
    --moe-ffn-hidden-size 768 \
    --moe-router-topk 8 \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff "${AUX_LOSS_COEFF:-1e-3}" \
    --moe-token-dispatcher-type alltoall \
    --moe-permute-fusion \
    --expert-model-parallel-size "${EP}" \
    --expert-tensor-parallel-size "${ETP}" \
    ${MOE_ARGS} \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --context-parallel-size "${CP}" \
    --micro-batch-size "${MBS}" \
    --global-batch-size "${GBS}" \
    --train-iters "${TRAIN_STEPS}" \
    --lr "${LR}" \
    --min-lr "${MIN_LR}" \
    --lr-decay-style cosine \
    --lr-warmup-iters "${LR_WARMUP_ITERS}" \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer \
    "${OVERLAP_ARGS[@]}" \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${TOKENIZER_PATH}" \
    --train-data-path "${DATA_PATH}" \
    --valid-data-path "${DATA_PATH}" \
    --test-data-path "${DATA_PATH}" \
    --dataloader-type cyclic \
    --no-data-sharding \
    --split 98,1,1 \
    --num-workers 2 \
    --eval-interval 1000000 \
    --eval-iters 1 \
    --save-interval 1000000 \
    --log-interval 1 \
    --log-throughput \
    "${LOAD_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo "Log: ${LOG_FILE}"
