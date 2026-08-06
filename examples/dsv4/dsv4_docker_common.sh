#!/usr/bin/env bash
# Shared Docker/bootstrap helpers for run_dsv4.sh (source, do not execute).

dsv4_docker_bootstrap_setup() {
    USE_BOOTSTRAP=0
    BOOTSTRAP_MOUNT="${BOOTSTRAP_DIR}"
    if [[ "${IMAGE}" == "lumen/tests:latest" || "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        USE_BOOTSTRAP=1
        if [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
            BOOTSTRAP_MOUNT=""
        elif [[ ! -f "${BOOTSTRAP_DIR}/.ready" ]]; then
            echo "[prepare] bootstrap missing — running prepare_bootstrap.sh"
            bash "${SCRIPT_DIR}/prepare_bootstrap.sh"
        fi
    fi
}

dsv4_docker_ensure_image() {
    if docker image inspect "${IMAGE}" &>/dev/null; then
        return 0
    fi
    if [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        bash "${SCRIPT_DIR}/build_dsv4_lumen_image.sh"
    else
        echo "[ERROR] Image not found: ${IMAGE}"
        exit 1
    fi
}

dsv4_docker_check_rocm() {
    if ! ls /dev/kfd &>/dev/null; then
        echo "[ERROR] /dev/kfd not found — ROCm device not accessible."
        exit 1
    fi
}

dsv4_docker_mkdirs() {
    mkdir -p "${MODEL_DIR}" "${LOG_DIR}" "${MODEL_DIR}/miopen-cache" "${TVM_CACHE_DIR}" \
        "${DATA_DIR}" "${PIP_CACHE_DIR}"
}

dsv4_docker_build_mounts() {
    DOCKER_MOUNTS=(
        -v "${LUMEN_DIR}:/workspace/Lumen"
        -v "${MODEL_DIR}:/root/models"
        -v "${DATA_DIR}:/root/datasets"
        -v "${MODEL_DIR}/miopen-cache:/root/.config/miopen"
        -v "${TVM_CACHE_DIR}:/root/.cache/tvm-ffi"
        -v "${PIP_CACHE_DIR}:/root/.cache/pip"
    )
    if [[ "${DSV4_PROFILE}" == "flash" && -d "${NFS_ROOT}" ]]; then
        DOCKER_MOUNTS+=(-v "${NFS_ROOT}:${NFS_ROOT}")
    fi
    if [[ -d "${TILEKERNELS_DIR}" ]]; then
        DOCKER_MOUNTS+=(-v "${TILEKERNELS_DIR}:/workspace/TileKernels")
    fi
    if [[ "${USE_BOOTSTRAP}" -eq 1 && -n "${BOOTSTRAP_MOUNT}" ]]; then
        DOCKER_MOUNTS+=(-v "${BOOTSTRAP_MOUNT}:/bootstrap:ro")
    fi
    if [[ "${DSV4_PROFILE}" == "flash" && -d /dev/infiniband ]]; then
        DOCKER_MOUNTS+=(--volume /dev/infiniband:/dev/infiniband)
    fi
}

dsv4_docker_append_bootstrap_env() {
    if [[ "${USE_BOOTSTRAP}" -eq 1 && -n "${BOOTSTRAP_MOUNT}" ]]; then
        DOCKER_ENV+=(-e BOOTSTRAP_DIR=/bootstrap)
    elif [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        DOCKER_ENV+=(-e BOOTSTRAP_DIR=/opt/dsv4-bootstrap -e WRITABLE_ROOT=/opt/dsv4-runtime)
    fi
}

dsv4_docker_append_kernel_env() {
    DOCKER_ENV+=(
        -e V4_SPARSE_MLA_BACKEND="${V4_SPARSE_MLA_BACKEND}"
        -e MHC_BACKEND="${MHC_BACKEND}"
        -e V4_INDEXER_IMPL="${V4_INDEXER_IMPL}"
        -e V4_INDEXER_BLOCK_N="${V4_INDEXER_BLOCK_N}"
        -e V4_INDEXER_NUM_STAGES="${V4_INDEXER_NUM_STAGES}"
        -e DSV4_ENABLE_RECOMPUTE="${DSV4_ENABLE_RECOMPUTE:-1}"
    )
    if [[ -d "${TILEKERNELS_DIR}" ]]; then
        DOCKER_ENV+=(-e TILEKERNELS_DIR=/workspace/TileKernels)
    fi
}

dsv4_docker_append_rocm_env() {
    DOCKER_ENV+=(
        -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        -e CUDA_DEVICE_MAX_CONNECTIONS=1
        -e NCCL_NVLS_ENABLE=0
        -e RCCL_MSCCL_ENABLE=0
        -e HSA_FORCE_FINE_GRAIN_PCIE=1
        -e TORCHDYNAMO_DISABLE=1
    )
}

dsv4_docker_append_multinode_env() {
    DOCKER_ENV+=(
        -e NNODES="${NNODES}"
        -e NPROC_PER_NODE="${NPROC_PER_NODE}"
        -e NODE_RANK="${NODE_RANK}"
        -e MASTER_ADDR="${MASTER_ADDR}"
        -e MASTER_PORT="${MASTER_PORT}"
        -e OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION}"
        -e DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES}"
        -e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
        -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens14np0}"
        -e NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
        -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
        -e MEGATRON_PATH="${MEGATRON_PATH}"
    )
}

dsv4_docker_run_opts_single() {
    DOCKER_RUN=(
        docker run --rm
        --name "${CONTAINER_NAME}"
        --device /dev/kfd
        --device /dev/dri
        --group-add video
        --group-add render
        --ipc=host
        --network=host
        --shm-size=128g
        --cap-add=SYS_PTRACE
        --security-opt seccomp=unconfined
        --ulimit memlock=-1
    )
}

dsv4_docker_run_opts_multinode() {
    DOCKER_RUN=(
        docker run --rm
        --name "${CONTAINER_NAME}"
        --device /dev/kfd
        --device /dev/dri
        --group-add video
        --ipc=host
        --network=host
        --shm-size=128g
        --cap-add=SYS_PTRACE
        --security-opt seccomp=unconfined
        --ulimit memlock=-1
        --ulimit nofile=65536:524288
        --group-add render
    )
}
