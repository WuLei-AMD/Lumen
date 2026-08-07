import torch

from tile_kernels.quant.cast_back_kernel import per_token_cast_back
from tile_kernels.quant.per_token_cast_kernel import per_token_cast


def fp8_simulate(x: torch.Tensor, block_size: int):
    """Simulate per-token FP8 cast + dequant using TileKernels quant helpers."""
    x_c = x.contiguous()
    n = x_c.size(-1)
    y_flat, scale_flat = per_token_cast(
        x_c.view(-1, n),
        "e4m3",
        block_size,
        round_sf=True,
        use_packed_ue8m0=True,
    )
    out_dtype = "bf16" if x.dtype == torch.bfloat16 else "fp32"
    out_flat = per_token_cast_back((y_flat, scale_flat), out_dtype, block_size)
    return out_flat.view_as(x_c).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
