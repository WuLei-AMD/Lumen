###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Tensor layout helpers for expert-major MoE dispatch."""

from collections.abc import Sequence
from typing import Literal

import torch


def transpose_variable_chunks(
    tensor: torch.Tensor,
    counts: torch.Tensor | Sequence[Sequence[int]],
    *,
    source_layout: Literal["sender_major", "expert_major"],
    fused: bool = False,
) -> torch.Tensor:
    """Transpose variable token chunks between sender-major and expert-major.

    ``counts[sender][expert]`` describes each chunk length. Sender-major input
    is laid out as ``[s0e0, s0e1, ..., s1e0, ...]``; expert-major input uses
    ``[e0s0, e0s1, ..., e1s0, ...]``. The operation is differentiable with
    respect to ``tensor``.
    """
    count_tensor = counts if isinstance(counts, torch.Tensor) else None
    if count_tensor is not None:
        if count_tensor.dim() != 2:
            raise ValueError("counts must be a sender-by-expert matrix")
        num_senders, num_experts = count_tensor.shape
        count_rows = None
    else:
        count_rows = counts
        num_senders = len(count_rows)
        num_experts = len(count_rows[0]) if num_senders else 0
    if num_senders == 0:
        return tensor
    if count_rows is not None and any(
        len(row) != num_experts for row in count_rows
    ):
        raise ValueError("counts must be a rectangular sender-by-expert matrix")
    if num_senders == 1 or num_experts == 1:
        return tensor

    if source_layout == "sender_major":
        input_pairs = [
            (sender, expert)
            for sender in range(num_senders)
            for expert in range(num_experts)
        ]
        output_pairs = [
            (sender, expert)
            for expert in range(num_experts)
            for sender in range(num_senders)
        ]
    elif source_layout == "expert_major":
        input_pairs = [
            (sender, expert)
            for expert in range(num_experts)
            for sender in range(num_senders)
        ]
        output_pairs = [
            (sender, expert)
            for sender in range(num_senders)
            for expert in range(num_experts)
        ]
    else:
        raise ValueError(f"Unsupported source_layout: {source_layout}")

    if fused and tensor.is_cuda:
        try:
            from transformer_engine.pytorch.permutation import (
                moe_sort_chunks_by_index,
            )
        except ImportError:
            moe_sort_chunks_by_index = None
        if moe_sort_chunks_by_index is not None:
            if count_tensor is None:
                count_tensor = torch.tensor(
                    count_rows,
                    dtype=torch.int32,
                    device=tensor.device,
                )
            else:
                count_tensor = count_tensor.to(device=tensor.device, dtype=torch.int32)
            if source_layout == "sender_major":
                split_sizes_tensor = count_tensor.reshape(-1)
                sorted_indices = (
                    torch.arange(
                        num_senders * num_experts,
                        dtype=torch.int32,
                        device=tensor.device,
                    )
                    .view(num_senders, num_experts)
                    .T.reshape(-1)
                )
            else:
                split_sizes_tensor = count_tensor.T.contiguous().reshape(-1)
                sorted_indices = (
                    torch.arange(
                        num_senders * num_experts,
                        dtype=torch.int32,
                        device=tensor.device,
                    )
                    .view(num_experts, num_senders)
                    .T.reshape(-1)
                )
            return moe_sort_chunks_by_index(
                tensor,
                split_sizes_tensor,
                sorted_indices,
            )

    if count_rows is None:
        count_rows = count_tensor.tolist()
    split_sizes = [
        int(count_rows[sender][expert]) for sender, expert in input_pairs
    ]
    if sum(split_sizes) != tensor.shape[0]:
        raise ValueError(
            f"Chunk counts sum to {sum(split_sizes)}, expected {tensor.shape[0]}"
        )
    chunks = tensor.split(split_sizes, dim=0)
    chunk_by_pair = dict(zip(input_pairs, chunks))
    return torch.cat([chunk_by_pair[pair] for pair in output_pairs], dim=0)
