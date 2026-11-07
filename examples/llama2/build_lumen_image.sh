#!/bin/bash
# Build a Docker image with the latest Lumen code baked in on top of the
# base lumen/llama2:latest image. Run from the Lumen repo root or from anywhere.
set -euo pipefail

LUMEN_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${1:-dev}"
IMAGE="lumen/llama2:${TAG}"

echo "[build] Context: ${LUMEN_DIR}"
echo "[build] Target:  ${IMAGE}"

docker build \
    --file "${LUMEN_DIR}/Dockerfile.lumen" \
    --tag  "${IMAGE}" \
    "${LUMEN_DIR}"

echo "[build] Done: ${IMAGE}"
echo "Run training with:"
echo "  IMAGE=${IMAGE} HOST_MODEL=/path/to/model HOST_DATA=/path/to/data bash examples/llama2/run_fsdp_lora_7b.sh"
