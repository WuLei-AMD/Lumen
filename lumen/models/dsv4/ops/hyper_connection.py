"""DeepSeek V4 Hyper-Connection utility backed by AIter.

Public API (`HCHeadParams`, `DeepSeekV4HyperConnectionUtil`) preserved so that
the Megatron-LM patch (radixark/Megatron-LM PR #28) call sites in
``transformer_layer.py`` and ``transformer_block.py`` keep working.

Internals route ``hc_pre_raw``/``hc_post_raw``/``hc_head_raw`` through
the fused DSV4 MHC APIs in AIter.
"""

import inspect
import os
from pathlib import Path

import einops
import torch
from aiter.ops.triton.fusions.mhc import mhc_head_dsv4, mhc_post_dsv4, mhc_pre_dsv4
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

# DeepSeek V4 uses post = 2 * sigmoid(...) for the post-layer mix.
_HC_POST_MULT_VALUE = 2.0


def _dsv4_mhc_dx_on() -> bool:
    return os.environ.get("DSV4_DUMP_MHC_DX", "0") == "1"


def _dsv4_mhc_dx_layer_sub(hc_fn: Tensor | None = None) -> tuple[int, str]:
    layer, sub = -1, "unk"
    for fr in inspect.stack()[1:14]:
        slf = fr.frame.f_locals.get("self")
        if slf is None or not hasattr(slf, "layer_number") or not hasattr(slf, "self_attention"):
            continue
        layer = int(slf.layer_number)
        if hc_fn is not None:
            if getattr(slf, "hc_attn_fn", None) is hc_fn:
                sub = "attn"
            elif getattr(slf, "hc_ffn_fn", None) is hc_fn:
                sub = "ffn"
        break
    return layer, sub


def _dsv4_write_dh_line(grad: Tensor, site: str, layer: int, sub: str) -> None:
    g = grad.detach().float()
    gsq = float(g.square().sum().item())
    try:
        from megatron.core import parallel_state as mpu

        tp = int(mpu.get_tensor_model_parallel_rank())
        pp = int(mpu.get_pipeline_model_parallel_rank())
    except Exception:
        tp, pp = -1, -1
    world = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    dump_dir = Path(os.environ.get("DSV4_DUMP_MHC_DX_DIR", "/root/models/dsv4_mhc_dx"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("DSV4_DUMP_TAG", "unk")
    path = dump_dir / f"{tag}_w{world:02d}_pp{pp}_tp{tp}.tsv"
    line = f"{layer}\t{sub}\t{site}\t{tuple(grad.shape)}\t{grad.numel()}\t{gsq:.6e}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


class _DumpDh(torch.autograd.Function):
    """Identity that records d(tensor) even inside activation recompute."""

    @staticmethod
    def forward(ctx, tensor, site, layer, sub):
        ctx.site = site
        ctx.layer = int(layer)
        ctx.sub = sub
        return tensor

    @staticmethod
    def backward(ctx, grad):
        if grad is not None:
            try:
                _dsv4_write_dh_line(grad, ctx.site, ctx.layer, ctx.sub)
            except Exception:
                pass
        return grad, None, None, None


def _dsv4_mark_dh(tensor: Tensor, site: str, layer: int, sub: str) -> Tensor:
    if not _dsv4_mhc_dx_on() or tensor is None or not torch.is_tensor(tensor):
        return tensor
    return _DumpDh.apply(tensor, site, layer, sub)


def _dsv4_hook_dh(tensor: Tensor, site: str, layer: int, sub: str) -> None:
    if not _dsv4_mhc_dx_on() or tensor is None or not torch.is_tensor(tensor):
        return
    if tensor.grad_fn is None and not tensor.requires_grad:
        return

    def _hook(grad: Tensor | None):
        if grad is None:
            return None
        try:
            _dsv4_write_dh_line(grad, site, layer, sub)
        except Exception:
            pass
        return None

    try:
        tensor.register_hook(_hook)
    except RuntimeError:
        pass


def _as_fp32(tensor: Tensor) -> Tensor:
    return tensor if tensor.dtype == torch.float32 else tensor.float()


class HCHeadParams(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        hc_mult = config.dsv4_hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = torch.nn.Parameter(torch.zeros(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = torch.nn.Parameter(torch.zeros(hc_mult, dtype=torch.float32))
        self.hc_head_scale = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        for p in [self.hc_head_fn, self.hc_head_base, self.hc_head_scale]:
            p._keep_fp32 = True

    def forward(self):
        raise NotImplementedError


class DeepSeekV4HyperConnectionUtil:
    """Hyper-Connection helper that delegates to AIter DSV4 MHC kernels."""

    def __init__(self, config: TransformerConfig):
        self.norm_eps = config.layernorm_epsilon
        self.hc_mult = config.dsv4_hc_mult
        self.hc_sinkhorn_iters = config.dsv4_hc_sinkhorn_iters
        self.hc_eps = config.dsv4_hc_eps

    def hc_pre_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``x`` is ``(B, S, hc_mult, hidden)``. Returns layer input + post/comb mixes.

        AIter consumes a flattened ``(B*S, hc_mult, hidden)`` boundary while
        this public helper preserves the Megatron-facing batch/sequence layout.
        """
        dtype = x.dtype
        batch, sequence, hc_mult, hidden = x.shape
        residual = (
            x if x.dtype == torch.bfloat16 else x.bfloat16()
        ).contiguous().reshape(batch * sequence, hc_mult, hidden)
        post, comb, layer_input = mhc_pre_dsv4(
            residual,
            _as_fp32(hc_fn).contiguous(),
            _as_fp32(hc_scale).contiguous(),
            _as_fp32(hc_base).contiguous(),
            rms_eps=self.norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=_HC_POST_MULT_VALUE,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return (
            layer_input.reshape(batch, sequence, hidden).to(dtype),
            post.reshape(batch, sequence, hc_mult, 1),
            comb.reshape(batch, sequence, hc_mult, hc_mult),
        )

    def hc_post_raw(
        self,
        x: Tensor,
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        """``x``: ``(B, S, hidden)``; ``residual``: ``(B, S, hc_mult, hidden)``.

        AIter receives all leading batch/sequence dimensions flattened to ``M``.
        """
        dtype = x.dtype
        batch, sequence, hidden = x.shape
        hc_mult = residual.shape[-2]
        layer_input = (
            x if x.dtype == torch.bfloat16 else x.bfloat16()
        ).contiguous().reshape(batch * sequence, hidden)
        residual_flat = (
            residual if residual.dtype == torch.bfloat16 else residual.bfloat16()
        ).contiguous().reshape(batch * sequence, hc_mult, hidden)
        out = mhc_post_dsv4(
            layer_input,
            residual_flat,
            _as_fp32(post).contiguous().reshape(batch * sequence, hc_mult, 1),
            _as_fp32(comb)
            .contiguous()
            .reshape(batch * sequence, hc_mult, hc_mult),
        )
        return out.reshape(batch, sequence, hc_mult, hidden).to(dtype)

    def hc_head_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        """``x``: ``(B, S, hc_mult, hidden)``. Returns ``(B, S, hidden)``."""
        dtype = x.dtype
        batch, sequence, hc_mult, hidden = x.shape
        residual = (
            x if x.dtype == torch.bfloat16 else x.bfloat16()
        ).contiguous().reshape(batch * sequence, hc_mult, hidden)
        layer_input = mhc_head_dsv4(
            residual,
            _as_fp32(hc_fn).contiguous(),
            _as_fp32(hc_scale).contiguous(),
            _as_fp32(hc_base).contiguous(),
            rms_eps=self.norm_eps,
            hc_pre_eps=self.hc_eps,
        )
        return layer_input.reshape(batch, sequence, hidden).to(dtype)

    def layer_pre(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        layer, sub = _dsv4_mhc_dx_layer_sub(hc_fn)
        hc_fn = _as_fp32(hc_fn)
        hc_scale = _as_fp32(hc_scale)
        hc_base = _as_fp32(hc_base)

        _dsv4_hook_dh(hidden_states, "before_mhc_pre", layer, sub)
        x = einops.rearrange(hidden_states, "s b hc d -> b s hc d")
        x, post, comb = self.hc_pre_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        hidden_states = einops.rearrange(x, "b s d -> s b d")
        _dsv4_hook_dh(hidden_states, "before_input_ln", layer, sub)
        return hidden_states, post, comb

    def layer_post(
        self,
        output_with_bias: Tensor | tuple[Tensor, Tensor | None],
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        if isinstance(output_with_bias, tuple):
            out, bias = output_with_bias
            assert bias is None
        else:
            out = output_with_bias
        assert isinstance(out, torch.Tensor)

        out = einops.rearrange(out, "s b d -> b s d")
        residual_bshd = einops.rearrange(residual, "s b hc d -> b s hc d")
        hidden_states = self.hc_post_raw(x=out, residual=residual_bshd, post=post, comb=comb)
        hidden_states = einops.rearrange(hidden_states, "b s hc d -> s b hc d")
        layer, sub = -1, "unk"
        for fr in inspect.stack()[1:12]:
            loc = fr.frame.f_locals
            slf = loc.get("self")
            if slf is not None and hasattr(slf, "layer_number") and hasattr(slf, "self_attention"):
                layer = int(slf.layer_number)
            if loc.get("hc_ffn_post") is post:
                sub = "ffn"
                break
            if loc.get("hc_attn_post") is post:
                sub = "attn"
                break
        _dsv4_hook_dh(hidden_states, "after_mhc_post", layer, sub)
        return hidden_states

    def block_expand(self, hidden_states: Tensor) -> Tensor:
        return einops.repeat(hidden_states, "s b d -> s b hc d", hc=self.hc_mult)

    def block_head(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        x = einops.rearrange(hidden_states, "s b hc d -> b s hc d")
        x = self.hc_head_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        return einops.rearrange(x, "b s d -> s b d")
