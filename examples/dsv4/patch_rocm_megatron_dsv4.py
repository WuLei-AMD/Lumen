#!/usr/bin/env python3
"""Add DSV4 support to ROCm/Megatron-LM (rocm_dev branch).

Thin CLI wrapper around ``examples/dsv4/megatron_patch`` modular patch steps.

Usage (inside container or on host):
    python3 examples/dsv4/patch_rocm_megatron_dsv4.py /path/to/Megatron-LM
"""

from __future__ import annotations

import sys
from pathlib import Path

_PATCH_DIR = Path(__file__).resolve().parent
if str(_PATCH_DIR) not in sys.path:
    sys.path.insert(0, str(_PATCH_DIR))

from megatron_patch import PATCH_VERSION, main  # noqa: E402


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <megatron-root>")
        print(f"Patch version: {PATCH_VERSION}")
        sys.exit(1)
    main(sys.argv[1])
