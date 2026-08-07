#!/usr/bin/env bash
# Shared helpers for DSV4 pretrain/profile inner scripts (source, do not execute).

dsv4_resolve_pretrain_ckpt() {
    local model_name="${1:?model_name}"
    local profile="${2:-4layer}"
    if [[ "${profile}" == "flash" ]]; then
        CKPT="/root/models/${model_name}_torch_dist"
        local fallback="/root/models/${model_name}_torch_dist_hc${DSV4_HC_MULT}"
    else
        CKPT="/root/models/${model_name}_torch_dist_hc${DSV4_HC_MULT}"
        local fallback="/root/models/${model_name}_torch_dist"
    fi
    if [[ ! -f "${CKPT}/latest_checkpointed_iteration.txt" && -f "${fallback}/latest_checkpointed_iteration.txt" ]]; then
        echo "[prepare] using fallback checkpoint ${fallback}"
        CKPT="${fallback}"
    fi
    export CKPT
}

dsv4_prepare_pretrain_ckpt() {
    local profile="${1:-4layer}"
    local skip_prepare="${2:-0}"
    if [[ "${skip_prepare}" == "1" || -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
        echo "[prepare] torch_dist checkpoint already present — skipping (path=${CKPT})"
        return 0
    fi
    if [[ ! -d /workspace/miles ]]; then
        echo "[ERROR] Checkpoint missing and MILES_DIR not mounted for prepare_dsv4_checkpoint.py"
        exit 1
    fi
    export PYTHONPATH="/workspace/Lumen:/workspace/miles:${PYTHONPATH:-}"
    export DSV4_PROFILE="${profile}"
    DSV4_HC_MULT="${DSV4_HC_MULT}" python examples/dsv4/prepare_dsv4_checkpoint.py
}

dsv4_pretrain_recompute_args() {
    RECOMPUTE_ARGS=()
    if [[ "${DSV4_ENABLE_RECOMPUTE:-1}" == "1" ]]; then
        RECOMPUTE_ARGS=(
            --recompute-granularity full
            --recompute-method uniform
            --recompute-num-layers 1
        )
    fi
    export RECOMPUTE_ARGS
}

dsv4_pretrain_load_args() {
    LOAD_ARGS=()
    if [[ "${LOAD_CKPT:-0}" == "1" && -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
        LOAD_ARGS=(--load "${CKPT}" --no-load-optim --no-load-rng)
        echo "[pretrain] loading checkpoint ${CKPT} (dsv4-hc-mult=${DSV4_HC_MULT})"
    else
        echo "[pretrain] training from random init (LOAD_CKPT=${LOAD_CKPT:-0})"
    fi
    export LOAD_ARGS
}
