#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME=${IMAGE_NAME:-lumen/qwen3-30b-a3b:latest}
BASE_IMAGE=${BASE_IMAGE:-rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1}

docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --tag "${IMAGE_NAME}" \
    --file "${SCRIPT_DIR}/Dockerfile" \
    "${REPO_ROOT}"
