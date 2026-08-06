#!/usr/bin/env bash
# Shared in-container bootstrap for DSV4 GRPO finetune (source, do not execute).

setup_dsv4_container_env() {
    if [[ -d /opt/dsv4-bootstrap && -f /opt/dsv4-bootstrap/.ready ]]; then
        unset MEGATRON_PATH PYTHONPATH
        export BOOTSTRAP_DIR=/opt/dsv4-bootstrap
        # shellcheck source=examples/dsv4/bootstrap_env.sh
        source examples/dsv4/bootstrap_env.sh
        export MEGATRON_PATH=/opt/dsv4-bootstrap/Megatron-LM
    elif [[ -d /bootstrap && -f /bootstrap/.ready ]]; then
        unset MEGATRON_PATH PYTHONPATH
        export BOOTSTRAP_DIR=/bootstrap
        # shellcheck source=examples/dsv4/bootstrap_env.sh
        source examples/dsv4/bootstrap_env.sh
        export MEGATRON_PATH=/bootstrap/Megatron-LM
    else
        export MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
    fi

    if [[ -f examples/dsv4/patch_rocm_megatron_dsv4.py && -d "${MEGATRON_PATH}" ]]; then
        echo "[setup] ensuring ROCm Megatron DSV4 patch on ${MEGATRON_PATH}"
        PYTHONPATH="/workspace/Lumen:${PYTHONPATH:-}" \
            python3 examples/dsv4/patch_rocm_megatron_dsv4.py "${MEGATRON_PATH}"
    fi

    if [[ -f examples/dsv4/patch_mi308x_tile_kernels.py && -n "${SITE_PKGS:-}" ]]; then
        TK_PATCH="${SITE_PKGS}/tile_kernels"
        if [[ -n "${TILEKERNELS_DIR:-}" && -d "${TILEKERNELS_DIR}/tile_kernels" ]]; then
            TK_PATCH="${TILEKERNELS_DIR}/tile_kernels"
        fi
        python3 examples/dsv4/patch_mi308x_tile_kernels.py "${TK_PATCH}"
    fi

    local datasets_dir="${MEGATRON_PATH}/megatron/core/datasets"
    if [[ -d "${datasets_dir}" ]] && ! compgen -G "${datasets_dir}/helpers_cpp*.so" >/dev/null; then
        echo "[setup] building Megatron helpers_cpp in ${datasets_dir} ..."
        make -C "${datasets_dir}" -j"$(nproc)"
    fi
}
