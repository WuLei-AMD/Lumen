"""Pure tensor mappings for Qwen3-MoE Hugging Face checkpoints."""

from __future__ import annotations

import torch


def pack_grouped_query_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> torch.Tensor:
    """Pack HF Q/K/V as Megatron's per-query-group ``[Q..., K, V]`` layout."""
    if num_attention_heads % num_query_groups:
        raise ValueError("num_attention_heads must be divisible by num_query_groups")
    if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("Q, K, and V weights must all be matrices")

    hidden_size = query.shape[1]
    queries_per_group = num_attention_heads // num_query_groups
    expected_q = (num_attention_heads * head_dim, hidden_size)
    expected_kv = (num_query_groups * head_dim, hidden_size)
    if tuple(query.shape) != expected_q:
        raise ValueError(f"unexpected Q shape {tuple(query.shape)}; expected {expected_q}")
    if tuple(key.shape) != expected_kv or tuple(value.shape) != expected_kv:
        raise ValueError(
            f"unexpected K/V shapes {tuple(key.shape)}/{tuple(value.shape)}; "
            f"expected {expected_kv}"
        )

    query = query.reshape(num_query_groups, queries_per_group * head_dim, hidden_size)
    key = key.reshape(num_query_groups, head_dim, hidden_size)
    value = value.reshape(num_query_groups, head_dim, hidden_size)
    return torch.cat((query, key, value), dim=1).reshape(-1, hidden_size)


def pack_swiglu_fc1(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Pack HF ``silu(gate) * up`` weights in Megatron SwiGLU order."""
    if gate.ndim != up.ndim or gate.shape != up.shape or gate.ndim not in (2, 3):
        raise ValueError(
            f"gate and up must be same-shaped matrices or expert stacks, got "
            f"{tuple(gate.shape)} and {tuple(up.shape)}"
        )
    return torch.cat((gate, up), dim=-2)


def expert_range_for_ep_rank(num_experts: int, ep_size: int, ep_rank: int) -> range:
    """Return the contiguous global expert range owned by an EP rank."""
    if ep_size <= 0 or num_experts % ep_size:
        raise ValueError("num_experts must be divisible by a positive ep_size")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank {ep_rank} is outside [0, {ep_size})")
    experts_per_rank = num_experts // ep_size
    start = ep_rank * experts_per_rank
    return range(start, start + experts_per_rank)
