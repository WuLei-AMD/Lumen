#!/usr/bin/env bash
# Reproducibility helpers for DSV4 native Megatron pretrain (source, do not execute).
#
# Optional env (host → container → torchrun):
#   SEED=42                 Megatron --seed (unset → Megatron default, usually 1234)
#   DETERMINISTIC=1         --deterministic-mode + NCCL/CUBLAS deterministic env
#   DSV4_MOCK_TOKEN_BASE=100  mock-data token base (see lumen.models.dsv4.megatron.pretrain)
#   DSV4_ROLLOUT_TOKENS_PATH=...  hash-safe rollout token cache (preferred when LOAD_CKPT=1)

DSV4_PRETRAIN_REPRO_ARGS=()

dsv4_pretrain_setup_repro() {
    DSV4_PRETRAIN_REPRO_ARGS=()
    export DSV4_MOCK_TOKEN_BASE="${DSV4_MOCK_TOKEN_BASE:-100}"
    export DSV4_ROLLOUT_TOKENS_PATH="${DSV4_ROLLOUT_TOKENS_PATH:-${MODEL_DIR:-/root/models}/dsv4_rollout_tokens.pt}"
    export DSV4_ALIGN_RL_ROUTING="${DSV4_ALIGN_RL_ROUTING:-1}"

    if [[ -n "${SEED:-}" ]]; then
        DSV4_PRETRAIN_REPRO_ARGS+=(--seed "${SEED}")
    fi

    if [[ "${DETERMINISTIC:-0}" == "1" ]]; then
        DSV4_PRETRAIN_REPRO_ARGS+=(--deterministic-mode)
        export NCCL_ALGO="${NCCL_ALGO:-Ring}"
        export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
        export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
    fi
}

dsv4_pretrain_print_repro() {
    echo "[repro] SEED=${SEED:-<megatron-default>} DETERMINISTIC=${DETERMINISTIC:-0} DSV4_MOCK_TOKEN_BASE=${DSV4_MOCK_TOKEN_BASE}"
    echo "[repro] DSV4_ROLLOUT_TOKENS_PATH=${DSV4_ROLLOUT_TOKENS_PATH:-unset} DSV4_ALIGN_RL_ROUTING=${DSV4_ALIGN_RL_ROUTING:-1}"
    if [[ "${DETERMINISTIC:-0}" == "1" ]]; then
        echo "[repro] NCCL_ALGO=${NCCL_ALGO} CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
    fi
}

dsv4_docker_append_pretrain_repro_env() {
    DOCKER_ENV+=(
        -e DETERMINISTIC="${DETERMINISTIC:-0}"
        -e DSV4_MOCK_TOKEN_BASE="${DSV4_MOCK_TOKEN_BASE:-100}"
        -e DSV4_ALIGN_RL_ROUTING="${DSV4_ALIGN_RL_ROUTING:-1}"
    )
    if [[ -n "${DSV4_ROLLOUT_TOKENS_PATH:-}" ]]; then
        DOCKER_ENV+=(-e "DSV4_ROLLOUT_TOKENS_PATH=${DSV4_ROLLOUT_TOKENS_PATH}")
    fi
    if [[ -n "${SEED:-}" ]]; then
        DOCKER_ENV+=(-e "SEED=${SEED}")
    fi
    if [[ "${DETERMINISTIC:-0}" == "1" ]]; then
        DOCKER_ENV+=(
            -e NCCL_ALGO="${NCCL_ALGO:-Ring}"
            -e NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
            -e CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
        )
    fi
}
