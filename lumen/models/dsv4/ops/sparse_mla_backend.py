"""Sparse MLA backend selection for DSV4 training."""

from __future__ import annotations

import os
from typing import Callable

import torch


def get_sparse_mla_backend() -> str:
    backend = os.environ.get("V4_SPARSE_MLA_BACKEND", "triton").lower()
    if backend != "triton":
        raise ValueError(f"V4_SPARSE_MLA_BACKEND={backend!r} is unsupported; use 'triton'")
    return backend


def get_sparse_attn_fn() -> Callable[..., torch.Tensor]:
    get_sparse_mla_backend()
    from lumen.models.dsv4.ops.kernel.triton_sparse_mla import sparse_attn_triton

    return sparse_attn_triton
