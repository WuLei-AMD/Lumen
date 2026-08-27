"""Sparse MLA for DSV4 training.

Default remains Triton. Set ``DSV4_FORCE_MLA_BACKEND=tilelang`` to use the
Miles-aligned TileLang kernels for a backward-numerics A/B.
``V4_SPARSE_MLA_BACKEND`` is intentionally ignored so existing tests stay locked
to Triton.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch


def get_sparse_mla_backend() -> str:
    forced = os.environ.get("DSV4_FORCE_MLA_BACKEND", "").strip().lower()
    if forced in ("tilelang", "triton"):
        return forced
    return "triton"


def get_sparse_attn_fn() -> Callable[..., torch.Tensor]:
    if get_sparse_mla_backend() == "tilelang":
        from lumen.kernels.dsv4.sparse_mla.tilelang_sparse_mla import (
            sparse_attn_tilelang,
        )

        print("[dsv4-mla] using TileLang sparse MLA (DSV4_FORCE_MLA_BACKEND=tilelang)", flush=True)
        return sparse_attn_tilelang
    from lumen.kernels.dsv4.sparse_mla.triton_sparse_mla import (
        sparse_attn_triton,
    )

    return sparse_attn_triton
