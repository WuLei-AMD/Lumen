###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""SonicMoE expert adapter for Megatron-Core MoE layers."""

from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn as nn

from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl


def _expert_weights(experts: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return FC1/FC2 weights in local-expert order."""
    if hasattr(experts, "linear_fc1") and hasattr(experts, "linear_fc2"):
        count = int(experts.num_local_experts)
        fc1 = [getattr(experts.linear_fc1, f"weight{i}") for i in range(count)]
        fc2 = [getattr(experts.linear_fc2, f"weight{i}") for i in range(count)]
        return fc1, fc2

    if hasattr(experts, "local_experts"):
        local_experts: Iterable[nn.Module] = experts.local_experts
        fc1 = [expert.linear_fc1.weight for expert in local_experts]
        fc2 = [expert.linear_fc2.weight for expert in local_experts]
        return fc1, fc2

    raise TypeError(f"Unsupported Megatron expert implementation: {type(experts).__name__}")


class SonicMoEExperts(nn.Module):
    """Megatron expert interface backed by AITER SonicMoE.

    Megatron dispatches and permutes token/expert assignments before invoking
    this module. SonicMoE's general-routing entry point consumes those
    assignments and performs both expert GEMMs, SwiGLU, routed-weight
    multiplication, and their backward passes.
    """

    def __init__(self, experts: nn.Module):
        super().__init__()
        self.config = experts.config
        self.num_local_experts = int(experts.num_local_experts)
        self.ep_group = getattr(experts, "ep_group", None)
        self.tp_group = getattr(experts, "tp_group", None)
        self.dp_group = getattr(experts, "dp_group", None)
        self.gemm_backend = os.environ.get("SONIC_MOE_GEMM_BACKEND", "triton")
        if self.gemm_backend not in {"triton", "blas"}:
            raise ValueError(
                "SONIC_MOE_GEMM_BACKEND must be either 'triton' or 'blas'"
            )

        if getattr(self.config, "add_bias_linear", False):
            raise ValueError("SonicMoE integration currently requires --disable-bias-linear")
        if not getattr(self.config, "gated_linear_unit", False):
            raise ValueError("SonicMoE integration currently requires a gated activation (SwiGLU)")
        expert_tp = getattr(self.config, "expert_tensor_parallel_size", 1)
        if expert_tp != 1:
            raise ValueError("SonicMoE integration currently requires expert tensor parallel size 1")

        fc1, fc2 = _expert_weights(experts)
        # Keep parameters physically expert-major. The zero-copy views passed
        # to SonicMoE below expose its [out_features, in_features, experts]
        # API without interleaving experts in storage.
        self.w1 = nn.Parameter(torch.stack([weight.detach() for weight in fc1], dim=0).contiguous())
        self.w2 = nn.Parameter(torch.stack([weight.detach() for weight in fc2], dim=0).contiguous())
        for parameter in self.parameters():
            parameter.allreduce = False

    def _forward_blas(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        tokens_per_expert,
    ) -> tuple[torch.Tensor, None]:
        """Run pre-routed local experts with rocBLAS-backed linear operations."""
        if isinstance(tokens_per_expert, torch.Tensor):
            split_sizes = tokens_per_expert.tolist()
        else:
            split_sizes = list(tokens_per_expert)

        hidden_chunks = torch.split(hidden_states, split_sizes)
        prob_chunks = torch.split(probs.reshape(-1), split_sizes)
        outputs = []
        for expert_id, (tokens, expert_probs) in enumerate(
            zip(hidden_chunks, prob_chunks)
        ):
            intermediate = torch.nn.functional.linear(tokens, self.w1[expert_id])
            activated = weighted_bias_swiglu_impl(
                intermediate,
                None,
                expert_probs.reshape(-1, 1),
                False,
            )
            output = torch.nn.functional.linear(activated, self.w2[expert_id])
            outputs.append(output)
        return torch.cat(outputs, dim=0), None

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert,
        permuted_probs: torch.Tensor,
    ):
        cpu_counts = tokens_per_expert
        if isinstance(tokens_per_expert, tuple):
            cpu_counts, gpu_counts = tokens_per_expert
            tokens_per_expert = gpu_counts if gpu_counts is not None else cpu_counts

        if self.gemm_backend == "blas":
            return self._forward_blas(
                permuted_local_hidden_states, permuted_probs, cpu_counts
            )

        device = permuted_local_hidden_states.device
        counts = torch.as_tensor(tokens_per_expert, dtype=torch.int32, device=device)

        from aiter.ops.triton.sonicmoe import (
            SonicMoEActivationType,
            moe_pre_routed_inputs,
        )

        output, _expert_frequency = moe_pre_routed_inputs(
            permuted_local_hidden_states,
            permuted_probs.reshape(-1).float(),
            counts,
            self.w1.permute(1, 2, 0),
            None,
            self.w2.permute(1, 2, 0),
            None,
            torch.cuda.current_stream().cuda_stream,
            SonicMoEActivationType.SWIGLU,
            False,
            True,
        )
        return output, None


def replace_megatron_moe_experts(model: nn.Module) -> int:
    """Replace Megatron ``MoELayer.experts`` modules with SonicMoE adapters."""
    replaced = 0
    for module in model.modules():
        if type(module).__name__ != "MoELayer" or not hasattr(module, "experts"):
            continue
        if isinstance(module.experts, SonicMoEExperts):
            continue
        module.experts = SonicMoEExperts(module.experts)
        replaced += 1
    return replaced


__all__ = ["SonicMoEExperts", "replace_megatron_moe_experts"]
