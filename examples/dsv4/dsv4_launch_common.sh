#!/usr/bin/env bash
# Shared helpers for 2-node launch_dsv4_*.sh scripts (source, do not execute).

# Fail launch when host Mem "used" exceeds this (GiB), e.g. leftover RAM from a prior
# crashed run with optimizer CPU offload (override or SKIP_HEALTH_CHECK=1).
: "${DSV4_LAUNCH_MAX_MEM_USED_GIB:=200}"
: "${DSV4_LAUNCH_MAX_GPU_USE_PCT:=5}"

dsv4_launch_init_ssh() {
    _ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
    if [[ -n "${SSH_KEY:-}" ]]; then
        _ssh_opts+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes)
    fi
}

dsv4_launch_validate_local_model_dirs() {
    if [[ -z "${MODEL_DIR:-}" ]]; then
        return 0
    fi
    if [[ "${MODEL_DIR}" == /nfs/* || "${MODEL_DIR}" == /mnt/data/* ]]; then
        echo "[launch][ERROR] MODEL_DIR must be local NVMe (got ${MODEL_DIR})"
        echo "  Head example: MODEL_DIR=/data1/\${USER}/models"
        exit 1
    fi
    local worker_dir="${WORKER_MODEL_DIR:-${MODEL_DIR}}"
    if [[ "${worker_dir}" == /nfs/* || "${worker_dir}" == /mnt/data/* ]]; then
        echo "[launch][ERROR] WORKER_MODEL_DIR must be local NVMe (got ${worker_dir})"
        echo "  Worker example: WORKER_MODEL_DIR=/mnt/nvme0n1/\${USER}/models"
        exit 1
    fi
}

dsv4_launch_cleanup_stale() {
    local container_names=("$@")
    local name=""
    for name in "${container_names[@]}"; do
        docker rm -f "${name}" 2>/dev/null || true
    done
    if [[ "${DSV4_LAUNCH_KILL_STALE_PROCS:-1}" == "1" ]]; then
        pkill -f 'finetune_dsv4_megatron|pretrain_dsv4_megatron' 2>/dev/null || true
    fi
}

dsv4_launch_mem_used_gib() {
    free -g | awk '/^Mem:/ {print $3; exit}'
}

dsv4_launch_check_host_health() {
    local tag="${1:-head}"
    local mem_used max_mem gpu_busy gpu_use

    mem_used="$(dsv4_launch_mem_used_gib)"
    max_mem="${DSV4_LAUNCH_MAX_MEM_USED_GIB}"
    echo "[health][${tag}] Mem used: ${mem_used} GiB (limit ${max_mem} GiB)"
    if [[ "${mem_used}" -gt "${max_mem}" ]]; then
        echo "[health][${tag}][ERROR] host memory too high (${mem_used} GiB > ${max_mem} GiB)."
        echo "  Wait for offload RAM to drop, or: docker rm -f lumen-dsv4-* ; pkill -f finetune_dsv4_megatron"
        echo "  Override: DSV4_LAUNCH_MAX_MEM_USED_GIB=${mem_used} or SKIP_HEALTH_CHECK=1"
        return 1
    fi

    if ! command -v rocm-smi >/dev/null 2>&1; then
        echo "[health][${tag}][WARN] rocm-smi not found — skip GPU idle check"
        return 0
    fi

    gpu_busy=0
    while IFS= read -r gpu_use; do
        [[ -z "${gpu_use}" ]] && continue
        if [[ "${gpu_use}" -gt "${DSV4_LAUNCH_MAX_GPU_USE_PCT}" ]]; then
            gpu_busy=1
            break
        fi
    done < <(rocm-smi --showuse 2>/dev/null | sed -n 's/.*GPU use (%): \([0-9][0-9]*\).*/\1/p')

    if [[ "${gpu_busy}" -eq 1 ]]; then
        echo "[health][${tag}][ERROR] one or more GPUs busy (>${DSV4_LAUNCH_MAX_GPU_USE_PCT}% use)."
        rocm-smi --showuse 2>/dev/null | grep 'GPU use' || true
        echo "  Stop other jobs or wait; override: SKIP_HEALTH_CHECK=1"
        return 1
    fi

    echo "[health][${tag}] GPUs idle (all <= ${DSV4_LAUNCH_MAX_GPU_USE_PCT}%)"
    return 0
}

dsv4_launch_check_worker_health() {
    local worker_ssh="${1:?worker_ssh}"
    ssh "${_ssh_opts[@]}" "${worker_ssh}" \
        "DSV4_LAUNCH_MAX_MEM_USED_GIB=${DSV4_LAUNCH_MAX_MEM_USED_GIB} \
DSV4_LAUNCH_MAX_GPU_USE_PCT=${DSV4_LAUNCH_MAX_GPU_USE_PCT} \
bash -s" <<'EOF'
set -euo pipefail
mem_used=$(free -g | awk '/^Mem:/ {print $3; exit}')
echo "[health][worker] Mem used: ${mem_used} GiB (limit ${DSV4_LAUNCH_MAX_MEM_USED_GIB} GiB)"
if [[ "${mem_used}" -gt "${DSV4_LAUNCH_MAX_MEM_USED_GIB}" ]]; then
    echo "[health][worker][ERROR] worker memory too high"
    exit 1
fi
if command -v rocm-smi >/dev/null 2>&1; then
    while IFS= read -r gpu_use; do
        [[ -z "${gpu_use}" ]] && continue
        if [[ "${gpu_use}" -gt "${DSV4_LAUNCH_MAX_GPU_USE_PCT}" ]]; then
            echo "[health][worker][ERROR] GPU busy: ${gpu_use}%"
            rocm-smi --showuse 2>/dev/null | grep 'GPU use' || true
            exit 1
        fi
    done < <(rocm-smi --showuse 2>/dev/null | sed -n 's/.*GPU use (%): \([0-9][0-9]*\).*/\1/p')
    echo "[health][worker] GPUs idle"
fi
EOF
}

dsv4_launch_run_health_checks() {
    local worker_ssh="${1:-}"
    if [[ "${SKIP_HEALTH_CHECK:-0}" == "1" ]]; then
        echo "[health] SKIP_HEALTH_CHECK=1 — skipping host health checks"
        return 0
    fi
    dsv4_launch_check_host_health head
    if [[ -n "${worker_ssh}" ]]; then
        dsv4_launch_check_worker_health "${worker_ssh}"
    fi
}

dsv4_launch_sync_worker_dsv4_scripts() {
    local worker_ssh="${1:?worker_ssh}"
    if [[ "${SKIP_WORKER_RSYNC:-0}" == "1" ]]; then
        echo "[launch] SKIP_WORKER_RSYNC=1 — skip examples/dsv4 sync"
        return 0
    fi
    echo "[launch] syncing examples/dsv4/ to worker (avoid head/worker arg skew) ..."
    rsync -a --delete \
        --exclude '__pycache__/' \
        --exclude '.gemm_tune/' \
        "${LUMEN_DIR}/examples/dsv4/" \
        "${worker_ssh}:${LUMEN_DIR}/examples/dsv4/" \
        -e "ssh ${_ssh_opts[*]}"
}

dsv4_launch_publish_preflight_id() {
    mkdir -p "${LOG_DIR}/.dsv4_preflight"
    echo "${PREFLIGHT_ID}" > "${LOG_DIR}/.dsv4_preflight/latest_id"
    echo "[launch] published PREFLIGHT_ID=${PREFLIGHT_ID} to NFS before node start"
}
