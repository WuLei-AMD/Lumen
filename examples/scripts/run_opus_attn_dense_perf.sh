#!/usr/bin/env bash
###############################################################################
# Run the six dense-pretrain jobs (llama2-7b / llama3-8b / qwen3-8b × bf16,fp8)
# against lumen:dev, then summarize Megatron iteration times.
#
# Usage (from anywhere):
#   bash examples/scripts/run_opus_attn_dense_perf.sh
#
# Env overrides:
#   IMAGE           default lumen:dev
#   LUMEN_DIR       default this repo
#   OUT_DIR         default ${LUMEN_DIR}/examples/results/opus_attn_dense_perf
#   TRAIN_STEPS     default 50 (passed through to each run_*.sh)
#   ATTN_BACKEND    default csrc  (passed as --lumen-attn-backend)
#   SKIP_WARMUP     default 2  (iters dropped from mean/median)
#   PARSE_ONLY=1    only re-parse existing logs
#   ONLY=llama2-7b:bf16,qwen3-8b:fp8   subset of jobs
###############################################################################
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
IMAGE="${IMAGE:-lumen:dev}"
OUT_DIR="${OUT_DIR:-${LUMEN_DIR}/examples/results/opus_attn_dense_perf}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
ATTN_BACKEND="${ATTN_BACKEND:-csrc}"
SKIP_WARMUP="${SKIP_WARMUP:-2}"
PARSE_ONLY="${PARSE_ONLY:-0}"
ONLY="${ONLY:-}"

export IMAGE LUMEN_DIR TRAIN_STEPS ATTN_BACKEND

mkdir -p "${OUT_DIR}"
SUITE_LOG="${OUT_DIR}/suite.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${SUITE_LOG}"; }

should_run() {
    local key="$1"
    if [[ -z "${ONLY}" ]]; then
        return 0
    fi
    [[ ",${ONLY}," == *",${key},"* ]]
}

run_one() {
    local key="$1"
    local model="$2"
    local prec="$3"
    local script="$4"
    local results_subdir="$5"

    if ! should_run "${key}"; then
        log "SKIP ${key} (not in ONLY=${ONLY})"
        return 0
    fi

    local results_dir="${OUT_DIR}/${results_subdir}"
    mkdir -p "${results_dir}"
    local job_log="${results_dir}/job_${prec}.outer.log"

    log "START ${key}  TRAIN_STEPS=${TRAIN_STEPS}  ATTN_BACKEND=${ATTN_BACKEND}  IMAGE=${IMAGE}"
    log "  script=${script}"
    log "  RESULTS_DIR=${results_dir}"

    local rc=0
    # Do not inherit set -e from caller; capture exit code.
    (
        cd "${LUMEN_DIR}"
        PRECISION="${prec}" \
        IMAGE="${IMAGE}" \
        LUMEN_DIR="${LUMEN_DIR}" \
        RESULTS_DIR="${results_dir}" \
        TRAIN_STEPS="${TRAIN_STEPS}" \
        ATTN_BACKEND="${ATTN_BACKEND}" \
        CONTAINER_NAME="lumen_opus_perf_${model}_${prec}_${ATTN_BACKEND}" \
            bash "${script}"
    ) >"${job_log}" 2>&1 || rc=$?

    if [[ ${rc} -eq 0 ]]; then
        log "OK    ${key}  (rc=0)  outer_log=${job_log}"
    else
        log "FAIL  ${key}  (rc=${rc})  outer_log=${job_log}"
        tail -n 40 "${job_log}" | tee -a "${SUITE_LOG}" || true
    fi
    return 0
}

if [[ "${PARSE_ONLY}" != "1" ]]; then
    log "==== suite start  repo=${LUMEN_DIR}  image=${IMAGE}  steps=${TRAIN_STEPS}  attn=${ATTN_BACKEND}"
    docker image inspect "${IMAGE}" >/dev/null || {
        log "ERROR: docker image ${IMAGE} not found"
        exit 1
    }

    run_one "llama2-7b:bf16" llama2_7b bf16 \
        "${LUMEN_DIR}/examples/llama2/run_pretrain_llama2_7b.sh" llama2
    run_one "llama2-7b:fp8" llama2_7b fp8 \
        "${LUMEN_DIR}/examples/llama2/run_pretrain_llama2_7b.sh" llama2
    run_one "llama3-8b:bf16" llama31_8b bf16 \
        "${LUMEN_DIR}/examples/llama31/run_pretrain_llama31_8b.sh" llama31
    run_one "llama3-8b:fp8" llama31_8b fp8 \
        "${LUMEN_DIR}/examples/llama31/run_pretrain_llama31_8b.sh" llama31
    run_one "qwen3-8b:bf16" qwen3_8b bf16 \
        "${LUMEN_DIR}/examples/qwen3/run_pretrain_qwen3_8b.sh" qwen3
    run_one "qwen3-8b:fp8" qwen3_8b fp8 \
        "${LUMEN_DIR}/examples/qwen3/run_pretrain_qwen3_8b.sh" qwen3

    log "==== suite jobs finished, parsing logs"
fi

python3 "${SCRIPT_DIR}/parse_pretrain_perf.py" \
    --out-dir "${OUT_DIR}" \
    --skip-warmup "${SKIP_WARMUP}" | tee -a "${SUITE_LOG}"

log "summary: ${OUT_DIR}/SUMMARY.txt"
log "csv:     ${OUT_DIR}/SUMMARY.csv"
