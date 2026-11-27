###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""SonicMoE drop-in for transformers-5.x fused Qwen3-MoE experts.

``Qwen3MoeSparseMoeBlock`` calls ``experts(hidden, top_k_index, top_k_weights)``
on a ``Qwen3MoeExperts`` holding ``gate_up_proj`` (E, 2I, H) and ``down_proj``
(E, H, I). :class:`SonicHFExperts` keeps that call signature, so the HF gate and
anything patched onto it (router precision, routing replay) stay in place, and
runs the experts as one SonicMoE grouped GEMM pair over all tokens.

The parameters are stored in SonicMoE's layout instead: ``w1`` (E, H, 2I) with
gate/up interleaved per column, and ``w2`` (E, I, H). :func:`hf_expert_state`
maps a full ``w1``/``w2`` back to the HF name and layout for weight export.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

_W1 = "w1"
_W2 = "w2"


def _dense(weight: torch.Tensor) -> torch.Tensor:
    """Unwrap FSDP2/DTensor views so Sonic grouped GEMM sees a dense tensor."""
    to_local = getattr(weight, "to_local", None)
    if not callable(to_local):
        return weight
    local = to_local()
    if tuple(local.shape) == tuple(weight.shape):
        return local
    return weight.full_tensor()


def _pack_gate_up(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=1)
    return torch.stack((gate, up), dim=2).flatten(1, 2).transpose(1, 2).contiguous()


def _unpack_gate_up(w1: torch.Tensor) -> torch.Tensor:
    num_experts, hidden, gate_up_dim = w1.shape
    pairs = w1.transpose(1, 2).reshape(num_experts, gate_up_dim // 2, 2, hidden)
    return torch.cat((pairs[:, :, 0], pairs[:, :, 1]), dim=1).contiguous()


def gate_up_to_sonic_(weight: torch.Tensor, expert_ids) -> None:
    """Rewrite HF ``gate_up_proj`` slices (2I, H) as Sonic ``w1`` (H, 2I), in place.

    The storage keeps its (E, 2I, H) shape; read it as Sonic through
    ``weight.view(E, H, 2I)``.
    """
    two_i, hidden = weight.shape[1:]
    for e in expert_ids:
        weight[e].view(hidden, two_i).copy_(_pack_gate_up(weight[e][None])[0])


def down_to_sonic_(weight: torch.Tensor, expert_ids) -> None:
    """Rewrite HF ``down_proj`` slices (H, I) as Sonic ``w2`` (I, H), in place."""
    hidden, inter = weight.shape[1:]
    for e in expert_ids:
        weight[e].view(inter, hidden).copy_(weight[e].t().contiguous())


def _fast_path_enabled() -> bool:
    return os.environ.get("LUMENRL_SONIC_FAST", "1") != "0"


def sonic_experts_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Routed experts on contiguous Sonic ``w1`` (E, H, 2I) and ``w2`` (E, I, H).

    Each token's output depends only on its own row and routing, for any token
    count, so a rollout engine calling this matches the trainer bitwise.

    Without autograd and for a small routed batch (decode) this takes the launch-lean
    path :func:`sonic_experts_forward_nograd`, which runs the same kernels on the same
    tiles and returns the same bits. ``LUMENRL_SONIC_FAST=0`` turns it off.
    """
    if hidden_states.shape[0] == 0:
        return torch.zeros_like(hidden_states)
    if _fast_path_enabled() and hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16:
        from lumen.ops.moe.route_layout import MAX_FAST_ROUTES

        needs_grad = torch.is_grad_enabled() and (
            hidden_states.requires_grad or w1.requires_grad or w2.requires_grad
        )
        if not needs_grad and top_k_index.numel() <= MAX_FAST_ROUTES:
            return sonic_experts_forward_nograd(hidden_states, top_k_index, top_k_weights, w1, w2)
    return _sonic_experts_forward_ref(hidden_states, top_k_index, top_k_weights, w1, w2)


def sonic_experts_forward_nograd(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Inference-only :func:`sonic_experts_forward` with far fewer launches.

    Same arithmetic as the reference path, kernel for kernel: one layout kernel gives the
    stable expert order (integer, exact), then the same row gather, grouped GEMMs, SwiGLU
    and gather-sum run directly, without the autograd wrappers, the extra arange/ones
    buffers, or the identity ``x * 1.0`` copy the reference puts after the down GEMM.
    """
    import triton
    from aiter.ops.triton.moe import sonicmoe

    from lumen.ops.moe.dispatch_layout import _BLOCK, _token_gather_kernels, _token_reduce_kernels
    from lumen.ops.moe.route_layout import route_layout

    tokens, hidden = hidden_states.shape
    topk = top_k_index.shape[1]
    num_experts = w1.shape[0]
    inter = w2.shape[1]
    rows, order, offsets = route_layout(top_k_index.reshape(-1), num_experts)
    slots = rows.numel()

    hidden_states = hidden_states.contiguous()
    routed = torch.empty(slots, hidden, dtype=hidden_states.dtype, device=hidden_states.device)
    gather_fwd, _ = _token_gather_kernels()
    gather_fwd[(slots, triton.cdiv(hidden, _BLOCK))](
        hidden_states,
        order,
        routed,
        hidden_states.stride(0),
        routed.stride(0),
        hidden,
        topk,
        _BLOCK,
    )

    h = torch.empty(slots, w1.shape[2], dtype=routed.dtype, device=routed.device)
    sonicmoe.grouped_gemm(routed, w1, offsets, out=h, bias=None, A_idx=None)
    a = sonicmoe.activation_fwd(h, inter, "swiglu", False)
    y = torch.empty(slots, hidden, dtype=a.dtype, device=a.device)
    sonicmoe.grouped_gemm(a, w2, offsets, out=y, bias=None)

    weights = top_k_weights.contiguous()
    out = torch.empty(tokens, hidden, dtype=y.dtype, device=y.device)
    reduce_fwd = _token_reduce_kernels()[0]
    reduce_fwd[(tokens, triton.cdiv(hidden, _BLOCK))](
        y,
        rows,
        weights,
        out,
        y.stride(0),
        out.stride(0),
        hidden,
        topk,
        _BLOCK,
    )
    return out


def _sonic_experts_forward_ref(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """The autograd-capable path: stable argsort layout, then Sonic up/down projections."""
    from aiter.ops.triton.moe import sonicmoe

    from lumen.ops.moe.dispatch_layout import gather_tokens, weighted_token_reduce

    num_experts = w1.shape[0]
    flat_experts = top_k_index.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    # bincount reads max().item(); histc fills the same bins without a sync.
    counts = torch.histc(
        flat_experts.to(torch.float32),
        bins=num_experts,
        min=0,
        max=num_experts - 1,
    ).to(torch.int32)
    routed, rows = gather_tokens(hidden_states, order, top_k_index.shape[1])
    expert_out = _pre_routed_swiglu(sonicmoe, routed, counts, w1, w2)
    return weighted_token_reduce(expert_out, rows, top_k_weights)


def _pre_routed_swiglu(sonicmoe, routed, counts, w1, w2):
    """``sonicmoe.moe_pre_routed_inputs`` with unit router scores and no host sync.

    The aiter entry checks ``counts.sum().item()`` against the row count, which
    breaks CUDA-graph capture of a rollout decode step. ``counts`` here is the
    histogram of the very rows being routed, so the check cannot fail.
    """
    T = routed.shape[0]
    device = routed.device
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), counts.cumsum(0, dtype=torch.int32)))
    identity = torch.arange(T, dtype=torch.int32, device=device)
    swiglu = sonicmoe.SonicMoEActivationType.SWIGLU
    a, h = sonicmoe._UpProjection.apply(
        routed, w1, None, offsets, T, 1, identity, identity, identity,
        None, False, swiglu, False, False, True, True,
    )
    scores = torch.ones(T, 1, dtype=torch.float32, device=device)
    return sonicmoe._DownProjection.apply(
        a, h, w2, None, scores, offsets, T, 1, identity, identity, identity,
        None, False, swiglu, True, False,
    )


class SonicHFExperts(nn.Module):
    """``Qwen3MoeExperts`` replacement backed by AITER SonicMoE (BF16)."""

    def __init__(self, experts: nn.Module, device: Optional[torch.device] = None):
        super().__init__()
        gate_up = experts.gate_up_proj.detach()
        down = experts.down_proj.detach()
        if device is not None:
            gate_up = gate_up.to(device)
            down = down.to(device)
        self.num_experts = int(gate_up.shape[0])
        self.w1 = nn.Parameter(_pack_gate_up(gate_up))
        self.w2 = nn.Parameter(down.transpose(1, 2).contiguous())

        if os.environ.get("SONIC_MOE_GEMM_BACKEND", "triton") != "triton":
            raise ValueError("SonicHFExperts supports SONIC_MOE_GEMM_BACKEND=triton only")
        from lumen.ops.moe.sonic_configs import install_qwen3_sonic_configs

        install_qwen3_sonic_configs()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return sonic_experts_forward(
            hidden_states, top_k_index, top_k_weights, _dense(self.w1), _dense(self.w2)
        )


def replace_hf_moe_experts(model: nn.Module, device: Optional[torch.device] = None) -> int:
    """Swap every fused HF ``mlp.experts`` for :class:`SonicHFExperts`. Returns #replaced."""
    replaced = 0
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None or isinstance(experts, SonicHFExperts):
            continue
        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            continue
        if not hasattr(module, "gate"):
            continue
        module.experts = SonicHFExperts(experts, device=device)
        replaced += 1
    return replaced


def hf_expert_state(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    """Map a full SonicHFExperts parameter to its HF name and layout; others pass through."""
    prefix, _, leaf = name.rpartition(".")
    if not prefix.endswith(".experts") or tensor.dim() != 3:
        return name, tensor
    if leaf == _W1:
        return f"{prefix}.gate_up_proj", _unpack_gate_up(tensor)
    if leaf == _W2:
        return f"{prefix}.down_proj", tensor.transpose(1, 2).contiguous()
    return name, tensor


def is_sonic_expert_param(name: str) -> bool:
    prefix, _, leaf = name.rpartition(".")
    return prefix.endswith(".experts") and leaf in (_W1, _W2)
