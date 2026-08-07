#!/usr/bin/env bash
# DSV4 mHC backend defaults — sourced from dsv4_paths.sh.
#
# Env:
#   MHC_BACKEND     triton (default) | tilelang
#   TILEKERNELS_DIR local TileKernels checkout (overlay in bootstrap_env.sh)

: "${MHC_BACKEND:=triton}"

if [[ "${MHC_BACKEND}" != "triton" && "${MHC_BACKEND}" != "tilelang" ]]; then
    echo "[dsv4_mhc] ERROR: MHC_BACKEND=${MHC_BACKEND} (use triton or tilelang)" >&2
    exit 1
fi

export MHC_BACKEND
