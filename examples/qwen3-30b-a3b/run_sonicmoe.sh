#!/usr/bin/env bash
# Qwen3-30B-A3B end-to-end SonicMoE training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOE_IMPL=sonic bash "${SCRIPT_DIR}/run_pretrain.sh"
