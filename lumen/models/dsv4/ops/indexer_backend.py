"""DSV4 DSA indexer backend selection (aiter)."""

from __future__ import annotations

import os
from typing import Callable

import torch


def get_indexer_impl() -> str:
    """Return configured indexer backend name (runtime kernel is aiter for both values)."""
    impl = os.environ.get("V4_INDEXER_IMPL", "aiter").lower()
    if impl not in ("aiter", "tilelang"):
        raise ValueError(f"V4_INDEXER_IMPL={impl!r} is unsupported; use 'aiter' or 'tilelang'")
    return impl


def get_batched_indexer_fwd() -> Callable[..., torch.Tensor]:
    get_indexer_impl()
    from lumen.models.dsv4.ops.kernel.aiter_dsv4_indexer import batched_indexer_fwd

    return batched_indexer_fwd


def get_batched_indexer_bwd() -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    get_indexer_impl()
    from lumen.models.dsv4.ops.kernel.aiter_dsv4_indexer import batched_indexer_bwd

    return batched_indexer_bwd
