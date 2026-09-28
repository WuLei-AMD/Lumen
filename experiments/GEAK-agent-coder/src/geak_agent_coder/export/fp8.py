"""Official-style 128x128 blockwise FP8 weight primitives."""

from __future__ import annotations

from dataclasses import dataclass

BLOCK_SIZE = 128
SCALE_SUFFIX = "_scale_inv"


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason: str


def weight_eligibility(name: str, tensor, *, block_size: int = BLOCK_SIZE) -> Eligibility:
    packed_expert = name.endswith((".experts.gate_up_proj", ".experts.down_proj"))
    if not name.endswith(".weight") and not packed_expert:
        return Eligibility(False, "not a weight key")
    expected_ndim = 3 if packed_expert else 2
    if getattr(tensor, "ndim", None) != expected_ndim:
        return Eligibility(False, "not a GEMM weight or packed expert stack")
    if not tensor.is_floating_point():
        return Eligibility(False, "not floating point")
    lowered = name.lower()
    skipped = (
        "embed",
        "lm_head",
        "norm",
        "router",
        ".gate.weight",
        "lora_",
    )
    if any(token in lowered for token in skipped):
        return Eligibility(False, "policy keeps embedding/head/norm/router/adapter in BF16")
    if tensor.shape[-2] % block_size or tensor.shape[-1] % block_size:
        return Eligibility(False, f"shape is not divisible by {block_size}x{block_size}")
    return Eligibility(True, "eligible GEMM matrix")


def scale_key_for(weight_key: str) -> str:
    if not weight_key.endswith(
        (".weight", ".experts.gate_up_proj", ".experts.down_proj")
    ):
        raise ValueError(f"expected a weight key, got {weight_key!r}")
    return f"{weight_key}{SCALE_SUFFIX}"


def _default_fp8_dtype(device):
    import torch

    # CPU PyTorch supports the OCP software conversion. Lumen/ROCm callers may
    # explicitly request FNUZ to match gfx942 kernels.
    return torch.float8_e4m3fn


def fp8_dtype_for_format(fp8_format: str):
    """Resolve an explicit manifest format to its PyTorch dtype."""

    import torch

    formats = {
        "e4m3fn": torch.float8_e4m3fn,
        "e4m3fnuz": torch.float8_e4m3fnuz,
    }
    try:
        return formats[fp8_format]
    except KeyError as exc:
        raise ValueError(
            f"unsupported FP8 format {fp8_format!r}; expected one of {sorted(formats)}"
        ) from exc


def fp8_format_for_dtype(dtype) -> str:
    """Return the exact OCP/FNUZ format represented by a PyTorch dtype."""

    import torch

    formats = {
        torch.float8_e4m3fn: "e4m3fn",
        torch.float8_e4m3fnuz: "e4m3fnuz",
    }
    try:
        return formats[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported FP8 dtype {dtype!r}") from exc


def quantize_blockwise_2d(
    weight,
    *,
    block_size: int = BLOCK_SIZE,
    fp8_dtype=None,
):
    """Quantize a 2-D tensor and return FP8 values plus dequant multipliers."""
    import torch

    if weight.ndim not in (2, 3):
        raise ValueError("blockwise2d quantization requires a matrix or expert stack")
    leading = weight.shape[:-2]
    rows, columns = weight.shape[-2:]
    if rows % block_size or columns % block_size:
        raise ValueError(f"shape {tuple(weight.shape)} is not divisible by {block_size}x{block_size}")
    fp8_dtype = fp8_dtype or _default_fp8_dtype(weight.device)
    fp8_max = float(torch.finfo(fp8_dtype).max)
    blocks = (
        weight.float()
        .reshape(*leading, rows // block_size, block_size, columns // block_size, block_size)
        .transpose(-3, -2)
    )
    amax = blocks.abs().amax(dim=(-1, -2))
    scale_inv = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
    quantized_blocks = (blocks / scale_inv[..., None, None]).clamp(-fp8_max, fp8_max)
    quantized = (
        quantized_blocks.to(fp8_dtype)
        .transpose(-3, -2)
        .reshape(*leading, rows, columns)
        .contiguous()
    )
    return quantized, scale_inv.float().contiguous()


def dequantize_blockwise_2d(
    quantized,
    scale_inv,
    *,
    block_size: int = BLOCK_SIZE,
    dtype=None,
):
    import torch

    if quantized.ndim not in (2, 3):
        raise ValueError("blockwise2d dequantization requires a matrix or expert stack")
    leading = quantized.shape[:-2]
    rows, columns = quantized.shape[-2:]
    expected = (*leading, rows // block_size, columns // block_size)
    if rows % block_size or columns % block_size or tuple(scale_inv.shape) != expected:
        raise ValueError(f"scale_inv shape {tuple(scale_inv.shape)} does not match {expected}")
    blocks = (
        quantized.float()
        .reshape(*leading, rows // block_size, block_size, columns // block_size, block_size)
        .transpose(-3, -2)
    )
    output = (
        (blocks * scale_inv.float()[..., None, None])
        .transpose(-3, -2)
        .reshape(*leading, rows, columns)
    )
    return output.to(dtype=dtype or torch.float32)


def quantization_error_metrics(
    reference,
    quantized,
    scale_inv,
    *,
    block_size: int = BLOCK_SIZE,
) -> dict[str, float | bool]:
    """Return finite-value and reconstruction metrics for one weight tensor."""

    import math
    import torch

    restored = dequantize_blockwise_2d(
        quantized, scale_inv, block_size=block_size
    )
    expected = reference.float()
    finite = bool(torch.isfinite(restored).all() and torch.isfinite(expected).all())
    error = restored - expected
    rmse = float(error.square().mean().sqrt())
    signal_rms = float(expected.square().mean().sqrt())
    snr_db = (
        float("inf")
        if rmse == 0.0
        else 20.0 * math.log10(max(signal_rms, 1.0e-30) / rmse)
    )
    return {
        "finite": finite,
        "rmse": rmse,
        "max_abs_error": float(error.abs().max()),
        "snr_db": snr_db,
    }
