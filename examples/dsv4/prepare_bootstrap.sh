#!/usr/bin/env bash
# Prepare DSV4 runtime bootstrap: local TileKernels + pip packages + ROCm Megatron-LM.
#
# Usage:
#   bash examples/dsv4/prepare_bootstrap.sh
#
# Output:
#   ${BOOTSTRAP_DIR}/Megatron-LM  -> symlink to ${MEGATRON_ROCM_DIR}
#   ${BOOTSTRAP_DIR}/site-packages/{tile_kernels,mbridge,...}
#
# Env (optional docker extract fallback):
#   BOOTSTRAP_EXTRACT_IMAGE     docker image with /opt/venv site-packages
#   BOOTSTRAP_EXTRACT_CONTAINER running container name (same layout)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"

CONTAINER="lumen-dsv4-bootstrap-extract-$$"
MEGATRON_STAMP="${BOOTSTRAP_DIR}/.megatron_rocm_ref"
SITE_PACKAGES="${BOOTSTRAP_DIR}/site-packages"
VENV_SITE="/opt/venv/lib/python3.10/site-packages"

_megatron_prepared() {
    [[ -f "${MEGATRON_STAMP}" ]] \
        && [[ "$(cat "${MEGATRON_STAMP}")" == "${MEGATRON_ROCM_REF}" ]] \
        && [[ -e "${BOOTSTRAP_DIR}/Megatron-LM/megatron/core/__init__.py" ]]
}

_prepare_rocm_megatron_bootstrap() {
    echo "[bootstrap] ROCm Megatron-LM (${MEGATRON_ROCM_REF})"
    bash "${SCRIPT_DIR}/prepare_rocm_megatron.sh"
    rm -rf "${BOOTSTRAP_DIR}/Megatron-LM"
    ln -sfn "${MEGATRON_ROCM_DIR}" "${BOOTSTRAP_DIR}/Megatron-LM"
    echo "${MEGATRON_ROCM_REF}" > "${MEGATRON_STAMP}"
}

_install_tile_kernels_local() {
    if [[ ! -d "${TILEKERNELS_DIR}/tile_kernels" ]]; then
        echo "[ERROR] TILEKERNELS_DIR missing tile_kernels package: ${TILEKERNELS_DIR}"
        echo "        Clone https://github.com/deepseek-ai/TileKernels next to Lumen or set TILEKERNELS_DIR."
        return 1
    fi
    echo "[bootstrap] tile_kernels from ${TILEKERNELS_DIR}"
    rm -rf "${SITE_PACKAGES}/tile_kernels"
    mkdir -p "${SITE_PACKAGES}/tile_kernels"
    rsync -a "${TILEKERNELS_DIR}/tile_kernels/" "${SITE_PACKAGES}/tile_kernels/"
    if [[ ! -f "${SITE_PACKAGES}/tile_kernels/_version.py" ]]; then
        echo '__version__ = "0.0.0+local"' > "${SITE_PACKAGES}/tile_kernels/_version.py"
    fi
}

_install_pip_packages() {
    echo "[bootstrap] pip packages -> ${SITE_PACKAGES}"
    mkdir -p "${SITE_PACKAGES}"
    python3 -m pip install \
        --target "${SITE_PACKAGES}" \
        --no-cache-dir --upgrade \
        mbridge

    # fast_hadamard_transform requires torch at build time; optional at runtime.
    if python3 -c "import torch" 2>/dev/null; then
        python3 -m pip install \
            --target "${SITE_PACKAGES}" \
            --no-cache-dir --upgrade \
            fast-hadamard-transform \
            || echo "[bootstrap] WARN: fast-hadamard-transform install failed (optional)"
    else
        echo "[bootstrap] skip fast-hadamard-transform on host (no torch; image build may add it)"
    fi
}

_resolve_extract_source() {
    local image="${BOOTSTRAP_EXTRACT_IMAGE:-}"
    if [[ -n "${BOOTSTRAP_EXTRACT_CONTAINER:-}" ]]; then
        echo "container:${BOOTSTRAP_EXTRACT_CONTAINER}"
        return 0
    fi
    if [[ -n "${image}" ]] && docker image inspect "${image}" &>/dev/null; then
        echo "image:${image}"
        return 0
    fi
    return 1
}

_copy_from_extract_source() {
    local pkg="$1"
    local source="$2"
    echo "  - site-packages/${pkg} (docker extract)"
    if [[ "${source}" == container:* ]]; then
        local cname="${source#container:}"
        docker cp "${cname}:${VENV_SITE}/${pkg}" "${SITE_PACKAGES}/${pkg}" \
            || echo "    (skip missing ${pkg})"
    else
        local image="${source#image:}"
        docker rm -f "${CONTAINER}" 2>/dev/null || true
        docker create --name "${CONTAINER}" "${image}" >/dev/null
        docker cp "${CONTAINER}:${VENV_SITE}/${pkg}" "${SITE_PACKAGES}/${pkg}" \
            || echo "    (skip missing ${pkg})"
        docker rm -f "${CONTAINER}" >/dev/null
    fi
}

_install_site_packages() {
    mkdir -p "${SITE_PACKAGES}"
    _install_tile_kernels_local || return 1
    _install_pip_packages

    local extract_source=""
    if extract_source="$(_resolve_extract_source)"; then
        echo "[bootstrap] optional docker overlay from ${extract_source}"
        for pkg in compressed_tensors ring_flash_attn torch_memory_saver tvm_ffi mooncake; do
            _copy_from_extract_source "${pkg}" "${extract_source}" || true
        done
        if [[ "${extract_source}" == image:* ]]; then
            mkdir -p "${BOOTSTRAP_DIR}/native-libs"
            docker rm -f "${CONTAINER}" 2>/dev/null || true
            docker create --name "${CONTAINER}" "${extract_source#image:}" >/dev/null
            docker cp "${CONTAINER}:/usr/local/lib/libetcd_wrapper.so" \
                "${BOOTSTRAP_DIR}/native-libs/" 2>/dev/null \
                || echo "    (skip missing libetcd_wrapper.so)"
            docker rm -f "${CONTAINER}" >/dev/null
        fi
    fi
}

if [[ -f "${BOOTSTRAP_DIR}/.ready" ]] && _megatron_prepared; then
    echo "[bootstrap] already prepared at ${BOOTSTRAP_DIR}"
    exit 0
fi

if [[ -f "${BOOTSTRAP_DIR}/.ready" ]]; then
    echo "[bootstrap] refreshing Megatron only (ROCm ref changed or missing)"
    _prepare_rocm_megatron_bootstrap
    echo "[bootstrap] Megatron refresh done: ${BOOTSTRAP_DIR}/Megatron-LM"
    exit 0
fi

_prepare_rocm_megatron_bootstrap
_install_site_packages

touch "${BOOTSTRAP_DIR}/.ready"
echo "[bootstrap] done: ${BOOTSTRAP_DIR}"
