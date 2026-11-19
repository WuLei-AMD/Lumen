###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Autograd-safe fused MoE router operations."""

from __future__ import annotations

from typing import Optional

import torch

from lumen.ops.dispatch import (
    _probe_aiter_softmax_topk,
    _probe_aiter_triton_moe_aux_loss,
)


def _aiter_softmax_topk(
    logits: torch.Tensor,
    topk: int,
    need_renorm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return full softmax scores and top-k results from AITER."""
    from aiter.ops.moe_op import softmax_topk

    logits_fp32 = logits.float().contiguous()
    num_tokens, num_experts = logits_fp32.shape
    scores = torch.empty_like(logits_fp32)
    topk_weights = torch.empty(
        num_tokens, topk, dtype=torch.float32, device=logits.device
    )
    topk_indices = torch.empty(
        num_tokens, topk, dtype=torch.int32, device=logits.device
    )
    token_expert_indices = torch.empty_like(topk_indices)
    softmax_topk(
        scores,
        topk_weights,
        topk_indices,
        token_expert_indices,
        logits_fp32,
        topk,
        need_renorm,
    )
    return scores, topk_weights, topk_indices.to(torch.int64)


def _softmax_topk(
    logits: torch.Tensor,
    topk: int,
    need_renorm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.is_cuda and _probe_aiter_softmax_topk():
        return _aiter_softmax_topk(logits, topk, need_renorm)

    scores = torch.softmax(logits.float(), dim=-1)
    topk_weights, topk_indices = scores.topk(topk, dim=-1)
    if need_renorm:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return scores, topk_weights, topk_indices


class _TopKRouter(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        topk: int,
        scaling_factor: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, topk_weights, topk_indices = _softmax_topk(logits, topk, True)
        base_probs = torch.zeros_like(logits, dtype=torch.float32)
        base_probs.scatter_(1, topk_indices, topk_weights)
        routing_map = base_probs.ne(0)
        scale = 1.0 if scaling_factor is None else float(scaling_factor)
        ctx.save_for_backward(base_probs)
        ctx.scale = scale
        ctx.input_dtype = logits.dtype
        return routing_map, base_probs * scale

    @staticmethod
    def backward(ctx, _grad_map, grad_probs: torch.Tensor):
        (base_probs,) = ctx.saved_tensors
        grad_base = grad_probs.float() * ctx.scale
        grad_logits = base_probs * (
            grad_base - (grad_base * base_probs).sum(dim=-1, keepdim=True)
        )
        return grad_logits.to(ctx.input_dtype), None, None


class _AuxScores(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores, _, topk_indices = _softmax_topk(logits, topk, False)
        routing_map = torch.zeros_like(scores, dtype=torch.bool)
        routing_map.scatter_(1, topk_indices, True)
        ctx.save_for_backward(scores)
        ctx.input_dtype = logits.dtype
        return routing_map, scores

    @staticmethod
    def backward(ctx, _grad_map, grad_scores: torch.Tensor):
        (scores,) = ctx.saved_tensors
        grad_scores = grad_scores.float()
        grad_logits = scores * (
            grad_scores - (grad_scores * scores).sum(dim=-1, keepdim=True)
        )
        return grad_logits.to(ctx.input_dtype), None


class _MoeAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        coefficient: float,
    ) -> torch.Tensor:
        coefficient = float(coefficient)
        ctx.save_for_backward(tokens_per_expert)
        ctx.coefficient = coefficient
        ctx.num_tokens, ctx.num_experts = probs.shape
        ctx.input_dtype = probs.dtype
        if probs.is_cuda and _probe_aiter_triton_moe_aux_loss():
            from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_fwd

            return moe_aux_loss_fwd(
                probs.float().contiguous(),
                tokens_per_expert.float().contiguous(),
                coefficient,
            )
        return (
            probs.float().sum(dim=0) * tokens_per_expert.float()
        ).sum() * coefficient

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (tokens_per_expert,) = ctx.saved_tensors
        if tokens_per_expert.is_cuda and _probe_aiter_triton_moe_aux_loss():
            from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_bwd

            grad_probs = moe_aux_loss_bwd(
                tokens_per_expert.float().contiguous(),
                ctx.coefficient,
                grad_output,
                ctx.num_tokens,
                ctx.num_experts,
            )
        else:
            grad_probs = (
                tokens_per_expert.float()
                .unsqueeze(0)
                .expand(ctx.num_tokens, -1)
                * (ctx.coefficient * grad_output)
            )
        return grad_probs.to(ctx.input_dtype), None, None


def fused_topk_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: Optional[int],
    group_topk: Optional[int],
    scaling_factor: Optional[float],
    score_function: str,
    expert_bias: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match Megatron's fused router contract using AITER softmax-topk."""
    if score_function != "softmax":
        raise NotImplementedError(
            f"score function {score_function!r} is unsupported; expected 'softmax'"
        )
    if num_groups is not None:
        raise NotImplementedError("Grouped routing is not supported")
    if group_topk is not None:
        raise NotImplementedError("Grouped top-k routing is not supported")
    if expert_bias is not None:
        raise NotImplementedError("expert_bias is not supported")
    # Softmax is monotonic, so selecting before or after it yields the same
    # experts; normalizing selected scores is softmax over selected logits.
    del use_pre_softmax
    return _TopKRouter.apply(logits, topk, scaling_factor)


def fused_compute_score_for_moe_aux_loss(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k routing map and full softmax scores for aux loss."""
    if score_function != "softmax":
        raise NotImplementedError(
            f"score function {score_function!r} is unsupported; expected 'softmax'"
        )
    return _AuxScores.apply(logits, topk)


def fused_moe_aux_loss(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    num_experts: int,
    topk: int,
    coefficient: float,
) -> torch.Tensor:
    """Compute Megatron switch load-balancing loss with fused kernels."""
    scaled_coefficient = (
        float(num_experts)
        * float(coefficient)
        / (float(topk) * float(total_num_tokens) * float(total_num_tokens))
    )
    return _MoeAuxLoss.apply(probs, tokens_per_expert, scaled_coefficient)
