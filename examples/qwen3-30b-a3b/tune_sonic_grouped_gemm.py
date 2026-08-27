#!/usr/bin/env python3
"""Offline-tune SonicMoE grouped GEMMs used by Qwen3-30B-A3B."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
    _grouped_gemm_dw_kernel,
    _grouped_gemm_kernel,
    grouped_gemm,
)


NUM_EXPERTS = 16
TOKENS = 32768
DTYPE = torch.bfloat16


def _offsets(device: torch.device) -> torch.Tensor:
    counts = torch.full(
        (NUM_EXPERTS,), TOKENS // NUM_EXPERTS, dtype=torch.int32, device=device
    )
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            counts.cumsum(0, dtype=torch.int32),
        )
    )


def _config_dict(config) -> dict:
    return {
        **config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def _new_configs(kernel, before: set) -> list[dict]:
    return [
        {"cache_key": repr(key), "config": _config_dict(kernel.cache[key])}
        for key in kernel.cache
        if key not in before
    ]


def main() -> None:
    device = torch.device("cuda")
    offsets = _offsets(device)
    identity = torch.arange(TOKENS, dtype=torch.int32, device=device)
    results = {}

    forward_specs = [
        ("up_forward", 2048, 1536, True),
        ("down_forward", 768, 2048, False),
        ("down_backward_input", 2048, 768, False),
        ("up_backward_input", 1536, 2048, False),
    ]
    for name, k_dim, n_dim, use_gather in forward_specs:
        a = torch.randn(TOKENS, k_dim, dtype=DTYPE, device=device)
        b = torch.randn(NUM_EXPERTS, k_dim, n_dim, dtype=DTYPE, device=device)
        before = set(_grouped_gemm_kernel.cache)
        grouped_gemm(a, b, offsets, A_idx=identity if use_gather else None)
        torch.cuda.synchronize()
        results[name] = {
            "kind": "forward",
            "K": k_dim,
            "N": n_dim,
            "E": NUM_EXPERTS,
            "has_gather": use_gather,
            "selected": _new_configs(_grouped_gemm_kernel, before),
        }
        del a, b
        torch.cuda.empty_cache()

    dw_specs = [
        ("dw1", 2048, 1536, True),
        ("dw2", 768, 2048, False),
    ]
    for name, k_dim, n_dim, use_gather in dw_specs:
        a = torch.randn(TOKENS, k_dim, dtype=DTYPE, device=device)
        b = torch.randn(TOKENS, n_dim, dtype=DTYPE, device=device)
        out = torch.empty(
            NUM_EXPERTS, k_dim, n_dim, dtype=DTYPE, device=device
        )
        before = set(_grouped_gemm_dw_kernel.cache)
        grouped_gemm(
            a,
            b,
            offsets,
            out=out,
            A_idx=identity if use_gather else None,
            A_is_transposed=True,
        )
        torch.cuda.synchronize()
        results[name] = {
            "kind": "dw",
            "K": k_dim,
            "N": n_dim,
            "E": NUM_EXPERTS,
            "has_gather": use_gather,
            "selected": _new_configs(_grouped_gemm_dw_kernel, before),
        }
        del a, b, out
        torch.cuda.empty_cache()

    output = Path(__file__).with_name("sonic_grouped_gemm_qwen3_tuned.json")
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output.read_text())


if __name__ == "__main__":
    main()
