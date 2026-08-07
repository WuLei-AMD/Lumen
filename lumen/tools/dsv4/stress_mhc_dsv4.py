#!/usr/bin/env python3
"""Stress AIter DSV4 MHC forward+backward at flash-training-like shapes."""

from __future__ import annotations

import argparse
import sys

import torch

from aiter.ops.triton.fusions.mhc import mhc_post_dsv4, mhc_pre_dsv4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--M", type=int, default=512, help="tokens per MHC call (B*S)")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--layers", type=int, default=43)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--sinkhorn", type=int, default=20)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP not available", file=sys.stderr)
        return 1

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    n = args.hc_mult
    rows = 2 * n + n * n
    fn = torch.randn(rows, n * args.hidden, device=device, dtype=torch.float32) * 0.03
    scale = torch.randn(3, device=device, dtype=torch.float32)
    base = torch.randn(rows, device=device, dtype=torch.float32) * 0.03

    print(
        f"device={device} M={args.M} hidden={args.hidden} n={n} "
        f"layers={args.layers} iters={args.iters} sinkhorn={args.sinkhorn} dtype={dtype}"
    )

    for it in range(1, args.iters + 1):
        stream = torch.randn(
            args.M, n, args.hidden, device=device, dtype=dtype, requires_grad=True
        )
        layer_input = stream
        for _layer in range(args.layers):
            post, comb, layer_input = mhc_pre_dsv4(
                stream, fn, scale, base, sinkhorn_repeat=args.sinkhorn
            )
            stream = mhc_post_dsv4(layer_input, stream, post, comb)

        loss = layer_input.float().sum()
        loss.backward()
        torch.cuda.synchronize()
        alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"iter {it}: loss={loss.item():.3f} max_alloc_gb={alloc:.2f}")
        torch.cuda.reset_peak_memory_stats(device)

    print("[ok] stress completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
