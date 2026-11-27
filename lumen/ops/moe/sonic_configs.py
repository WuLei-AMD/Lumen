"""Qwen3-30B-A3B SonicMoE grouped GEMM tiles for gfx950.

AITER ships SonicMoE configs only for gfx942 and reuses them on gfx950.
These tiles were tuned on MI350X for the Qwen3-30B-A3B blockwise FP8 expert
shapes at EP=8 (16 local experts) and are applied on top of AITER's pick for
FP8 calls when ``SONIC_MOE_USE_QWEN3_TUNED_GEMM=1``.
"""

import os
import threading

# (N, K, E, has_gather) -> tile
_QWEN3_FWD_CONFIGS = {
    (1536, 2048, 16, True): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
    # Pre-routed forward and dgrad have no gather index. BLOCK_K matches the
    # 128-wide FP8 scale block; eight warps were slower on these shapes.
    (1536, 2048, 16, False): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "num_warps": 4,
        "num_stages": 2,
    },
    (2048, 768, 16, False): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "num_warps": 4,
        "num_stages": 2,
    },
    (768, 2048, 16, False): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "num_warps": 4,
        "num_stages": 2,
    },
    (2048, 1536, 16, False): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "num_warps": 4,
        "num_stages": 2,
    },
}

# FP8 wgrad only: the BLOCK_T=128 tiles overflow LDS with BF16 operands.
_QWEN3_FP8_DW_CONFIGS = {
    (1536, 2048, 16, True): {
        "BLOCK_K": 128,
        "BLOCK_N": 128,
        "BLOCK_T": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
    # BLOCK_N=128 rereads each activation row half as often as 64 on the
    # 1536-wide w1 wgrad.
    (1536, 2048, 16, False): {
        "BLOCK_K": 128,
        "BLOCK_N": 128,
        "BLOCK_T": 128,
        "num_warps": 8,
        "num_stages": 2,
    },
    (2048, 768, 16, False): {
        "BLOCK_K": 128,
        "BLOCK_N": 128,
        "BLOCK_T": 128,
        "num_warps": 8,
        "num_stages": 2,
    },
}

_INSTALLED = False


def _tuned() -> bool:
    return os.environ.get("SONIC_MOE_USE_QWEN3_TUNED_GEMM", "0") == "1"


def qwen3_fp8_dw_config(base: dict, N: int, K: int, E: int, has_gather: bool) -> dict:
    """AITER's grouped wgrad tile with the Qwen3 FP8 override applied."""
    cfg = dict(base)
    if _tuned():
        cfg.update(_QWEN3_FP8_DW_CONFIGS.get((N, K, E, bool(has_gather)), {}))
    return cfg


def install_qwen3_sonic_configs() -> None:
    """Route AITER SonicMoE forward/dgrad tile lookups through the Qwen3 gfx950 tiles."""
    global _INSTALLED
    if _INSTALLED or not _tuned():
        return
    try:
        import aiter.ops.triton.moe.sonicmoe as sonicmoe
    except ImportError:
        return
    if not hasattr(sonicmoe, "get_grouped_gemm_fwd_config"):
        return

    aiter_fwd = sonicmoe.get_grouped_gemm_fwd_config
    aiter_triton = sonicmoe._grouped_gemm_triton
    fp8_call = threading.local()

    def _fwd(N, K, E, has_gather=False, use_specialized=False):
        cfg = dict(aiter_fwd(N, K, E, has_gather, use_specialized))
        if getattr(fp8_call, "active", False):
            cfg.update(_QWEN3_FWD_CONFIGS.get((N, K, E, bool(has_gather)), {}))
        return cfg

    # The tiles were tuned for FP8 operands; BF16 stays on AITER's pick.
    def _triton(*args, **kwargs):
        a_scale = kwargs.get("A_scale", args[8] if len(args) > 8 else None)
        fp8_call.active = a_scale is not None
        try:
            return aiter_triton(*args, **kwargs)
        finally:
            fp8_call.active = False

    sonicmoe.get_grouped_gemm_fwd_config = _fwd
    sonicmoe._grouped_gemm_triton = _triton
    _INSTALLED = True
