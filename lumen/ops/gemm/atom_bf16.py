"""BF16 kernels used by a Lumen training forward with ``rollout: ATOM``.

Qwen3-8B fused QKV is pinned to the gfx950 prefill ``split_k1`` kernel.
``o_proj`` uses ``F.linear``. Every other BF16 projection uses ``TunedGemm``.
"""

from __future__ import annotations

QKV_WEIGHT = (6144, 4096)
QKV_SOLIDX = 4696
QKV_KERNEL = (
    "flydsl_gemm5_abf16_wbf16_bf16_t64x64x128_split_k1_"
    "block_m_warp2_block_n_warp2_block_k_warp2_"
    "async_copyTrue_b_to_ldsTrue_b_preshuffleFalse_c_to_ldsFalse_gfx950"
)

o_proj_linear_count = 0


def _restore(activation, out, width: int):
    if activation.dim() < 3:
        return out
    return out.view(*activation.shape[:-1], width)


def qkv_gemm(activation, weight, bias=None):
    """Fused QKV on the training ``split_k1`` kernel. Other shapes use TunedGemm."""
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.tuned_gemm import flydsl_gemm, tgemm

    flat = activation.reshape(-1, activation.shape[-1]) if activation.dim() >= 3 else activation
    if get_gfx() != "gfx950" or tuple(weight.shape) != QKV_WEIGHT:
        out = tgemm.mm(flat, weight, bias, otype=flat.dtype)
        return _restore(activation, out, weight.shape[0])
    out = flydsl_gemm(
        flat,
        weight,
        QKV_SOLIDX,
        bias=bias,
        otype=flat.dtype,
        config={"kernelName": QKV_KERNEL},
    )
    return _restore(activation, out, weight.shape[0])


def dense_bf16_gemm(activation, weight, bias=None):
    """BF16 projection used by the ATOM-aligned training forward."""
    from aiter.tuned_gemm import tgemm

    return tgemm.mm(activation, weight, bias, otype=activation.dtype)


def o_proj_linear(activation, weight, bias=None):
    """``o_proj`` GEMM. The tuned ASM split-K kernel is not stable across launches."""
    global o_proj_linear_count
    import torch.nn.functional as F

    o_proj_linear_count += 1
    if activation.dim() >= 3:
        flat = activation.reshape(-1, activation.size(-1))
        out = F.linear(flat, weight, bias)
        return out.view(*activation.shape[:-1], weight.shape[0])
    return F.linear(activation, weight, bias)
