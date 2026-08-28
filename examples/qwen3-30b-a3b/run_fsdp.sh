#!/usr/bin/env bash
# Qwen3-30B-A3B FSDP2 accuracy run aligned with the Megatron launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=${CONFIG:-"${SCRIPT_DIR}/config_MI350X_1x8x1.sh"}
source "${CONFIG}"
EXPERT_BACKEND=${EXPERT_BACKEND:-te_grouped}
MODEL_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_ARGS=(--model-name-or-path "${MODEL_PATH}")
fi

case "${EXPERT_BACKEND}" in
    sequential|te_grouped|sonic) ;;
    *)
        echo "ERROR: EXPERT_BACKEND must be sequential, te_grouped, or sonic" >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname "${DATA_PATH}")" "${RESULTS_DIR}"
if [ ! -s "${DATA_PATH}" ]; then
    python3 - "${DATA_PATH}" "${SEQ_LEN}" "${GBS}" "${TRAIN_STEPS}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
seq_len, global_batch, steps = map(int, sys.argv[2:])
documents = max(128, global_batch * (steps + 4))
text = " ".join(["training"] * max(1024, seq_len))
with path.open("w") as output:
    for index in range(documents):
        output.write(json.dumps({"text": f"{index} {text}"}) + "\n")
PY
fi

RUN_SUFFIX=${RUN_SUFFIX:-}
LOG_FILE="${RESULTS_DIR}/qwen3-30b-a3b-fsdp-${EXPERT_BACKEND}${RUN_SUFFIX:+-${RUN_SUFFIX}}-bf16-seq${SEQ_LEN}-mbs${MBS}-gbs${GBS}.log"
DP=$((NGPU / EP))
if [ $((NGPU % EP)) -ne 0 ]; then
    echo "ERROR: NGPU=${NGPU} must be divisible by EP=${EP}" >&2
    exit 2
fi
GRAD_ACCUM=$((GBS / (MBS * DP)))
if [ $((GBS % (MBS * DP))) -ne 0 ]; then
    echo "ERROR: GBS=${GBS} must be divisible by MBS*DP=$((MBS * DP))" >&2
    exit 2
fi

torchrun \
    --nproc_per_node="${NGPU}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/pretrain_qwen3_30b_a3b_fsdp.py" \
    "${MODEL_ARGS[@]}" \
    --tokenizer-name-or-path "${TOKENIZER_PATH}" \
    --train-data-path "${DATA_PATH}" \
    --val-data-path "${DATA_PATH}" \
    --data-format pretrain \
    --mode bf16 \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings 4096 \
    --micro-batch-size "${MBS}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --max-steps "${TRAIN_STEPS}" \
    --lr "${LR}" \
    --min-lr "${MIN_LR}" \
    --lr-decay-style cosine \
    --lr-warmup-steps "${LR_WARMUP_ITERS}" \
    --weight-decay 0.1 \
    --max-grad-norm 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --aux-loss-coeff 1e-3 \
    --ep-size "${EP}" \
    --dp-size "${DP}" \
    --expert-backend "${EXPERT_BACKEND}" \
    --sharding full_shard \
    --no-gradient-checkpointing \
    --num-workers 2 \
    --log-interval 1 \
    --eval-interval 0 \
    --seed 1234 \
    2>&1 | tee "${LOG_FILE}"

echo "Log: ${LOG_FILE}"
