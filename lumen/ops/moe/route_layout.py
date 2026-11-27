###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""One-kernel expert layout for small routed batches (decode).

``sonic_experts_forward`` sorts the flat ``[token, slot]`` expert ids with a stable
argsort, builds per-expert counts and their offsets, and inverts the permutation. That
is about a dozen tiny launches per MoE layer. For a few hundred routes one Triton
program computes the same three integer arrays:

* ``rows[i]``     sorted position of route ``i`` (what ``argsort`` + scatter produce),
* ``order[p]``    route index at sorted position ``p`` (``argsort(stable=True)``),
* ``offsets[e]``  exclusive prefix of the per-expert counts, ``offsets[E] = total``.

All three are exact integer results, so they equal the unfused ones bit for bit.
"""

from __future__ import annotations

import torch

# routes (tokens * top_k) the one-program kernel handles; beyond it callers use the
# argsort path, whose launch cost is small next to the grouped GEMMs on a large batch.
MAX_FAST_ROUTES = 256

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _route_layout(
        experts,
        rows,
        order,
        offsets,
        n_routes,
        E: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        i = tl.arange(0, BLOCK_N)
        valid = i < n_routes
        e = tl.load(experts + i, mask=valid, other=0).to(tl.int32)
        ex = tl.arange(0, BLOCK_E)
        onehot = (e[:, None] == ex[None, :]) & valid[:, None]
        hot = onehot.to(tl.int32)
        counts = tl.sum(hot, axis=0)
        inclusive = tl.cumsum(counts, axis=0)
        exclusive = inclusive - counts
        # Routes of one expert keep their original order: the rank inside the expert
        # is the running count of earlier routes with the same id.
        running = tl.cumsum(hot, axis=0)
        within = tl.sum(tl.where(onehot, running, 0), axis=1) - 1
        base = tl.sum(tl.where(onehot, exclusive[None, :], 0), axis=1)
        rank = base + within
        tl.store(rows + i, rank, mask=valid)
        tl.store(order + rank, i.to(tl.int64), mask=valid)
        tl.store(offsets + 1 + ex, inclusive, mask=ex < E)
        zero = tl.zeros([1], dtype=tl.int32)
        tl.store(offsets + tl.arange(0, 1), zero)

    _KERNEL = _route_layout
    return _KERNEL


def route_layout(flat_experts: torch.Tensor, num_experts: int):
    """Return ``(rows int32 [N], order int64 [N], offsets int32 [E + 1])``.

    ``flat_experts`` is the flattened ``top_k_index`` (any integer dtype, ``N`` entries).
    """
    import triton

    n = flat_experts.numel()
    if n > MAX_FAST_ROUTES:
        raise ValueError(f"route_layout handles at most {MAX_FAST_ROUTES} routes, got {n}")
    device = flat_experts.device
    rows = torch.empty(n, dtype=torch.int32, device=device)
    order = torch.empty(n, dtype=torch.int64, device=device)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    _kernel()[(1,)](
        flat_experts,
        rows,
        order,
        offsets,
        n,
        E=num_experts,
        BLOCK_N=max(triton.next_power_of_2(n), 16),
        BLOCK_E=max(triton.next_power_of_2(num_experts), 16),
        num_warps=8,
    )
    return rows, order, offsets
