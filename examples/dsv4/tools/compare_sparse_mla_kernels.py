#!/usr/bin/env python3
"""Compare DSV4 sparse-MLA TileLang vs Triton (fwd + bwd).

Reproduces the training kernel gap: last-PP dump had matching dO / q / kv / topk
but dkv ~1.6x and dq-path ~4x until Lumen switched to TileLang.

Usage (inside Lumen image, 1 GPU):
  python examples/dsv4/tools/compare_sparse_mla_kernels.py --preset small
  python examples/dsv4/tools/compare_sparse_mla_kernels.py --preset flash
  bash examples/dsv4/tools/run_compare_sparse_mla_kernels.sh --preset small
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import torch

# Host checkout when launched via docker mounts.
_LUMEN = os.environ.get("LUMEN_DIR", "/workspace/Lumen")
if _LUMEN not in sys.path:
    sys.path.insert(0, _LUMEN)
_AITER = os.environ.get("AITER_DIR", "/workspace/aiter")
if _AITER not in sys.path:
    sys.path.insert(0, _AITER)


@dataclass
class Diff:
    name: str
    cosine: float
    rel: float
    max_abs: float
    mean_abs: float
    sumsq_a: float
    sumsq_b: float
    sumsq_ratio: float  # b/a


def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1).cpu()


def compute_diff(name: str, a: torch.Tensor, b: torch.Tensor) -> Diff:
    x, y = _flatten(a), _flatten(b)
    absd = (x - y).abs()
    xy = float((x * y).sum())
    x2 = float((x * x).sum())
    y2 = float((y * y).sum())
    denom = x2 + y2
    rel = (1.0 - 2.0 * xy / denom) if denom > 0 else 0.0
    cos = xy / (math.sqrt(x2) * math.sqrt(y2) + 1e-30)
    ratio = y2 / x2 if x2 > 0 else float("nan")
    return Diff(name, cos, rel, float(absd.max()), float(absd.mean()), x2, y2, ratio)


def fmt(d: Diff) -> str:
    flag = "OK" if d.cosine > 0.99 and 0.5 < d.sumsq_ratio < 2.0 else "GAP"
    if d.cosine > 0.999 and 0.9 < d.sumsq_ratio < 1.1:
        flag = "OK"
    return (
        f"  {d.name:10s}  cos={d.cosine:.6f}  rel={d.rel:.3e}  "
        f"max={d.max_abs:.3e}  mean={d.mean_abs:.3e}  "
        f"sumsq_B/A={d.sumsq_ratio:.4f}  [{flag}]"
    )


def ref_sparse_mla(q, kv, attn_sink, topk_idxs, sm_scale):
    """FP32 batched reference. q [B,S,H,D], kv [B,Skv,D], idx [B,S,topk], sink [H]."""
    qf, kvf = q.float(), kv.float()
    b, m, h, d = qf.shape
    n = kvf.shape[1]
    topk = topk_idxs.shape[-1]
    if sm_scale is None:
        sm_scale = d**-0.5

    attn_mask = torch.zeros(b, m, n, device=q.device, dtype=torch.bool)
    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, m, topk)
    seq_idx = torch.arange(m, device=q.device).view(1, m, 1).expand(b, m, topk)
    valid = topk_idxs != -1
    attn_mask[batch_idx[valid], seq_idx[valid], topk_idxs[valid].long()] = True

    scores = torch.einsum("bmhd,bnd->bmhn", qf, kvf) * sm_scale
    scores = scores.masked_fill(~attn_mask.unsqueeze(2).expand(-1, -1, h, -1), float("-inf"))
    scores_max = scores.max(dim=-1, keepdim=True).values.clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max)
    numerator = torch.einsum("bmhn,bnd->bmhd", exp_scores, kvf)
    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(attn_sink.view(1, 1, h) - scores_max.squeeze(-1))
    o = numerator / (sum_exp + sink_term).unsqueeze(-1)
    return o.to(q.dtype)


def make_inputs(batch, seqlen, heads, dim, seqlen_kv, topk, seed, device, n_invalid=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16, generator=g)
    kv = torch.randn(batch, seqlen_kv, dim, device=device, dtype=torch.bfloat16, generator=g)
    sink = torch.randn(heads, device=device, dtype=torch.float32, generator=g) * 0.1
    actual = min(topk, seqlen_kv)
    rows = []
    for _b in range(batch):
        per_s = []
        for _s in range(seqlen):
            perm = torch.randperm(seqlen_kv, device=device, generator=g)[:actual]
            if actual < topk:
                pad = torch.full((topk - actual,), -1, device=device, dtype=torch.int32)
                perm = torch.cat([perm.to(torch.int32), pad])
            else:
                perm = perm.to(torch.int32)
            per_s.append(perm)
        rows.append(torch.stack(per_s))
    idx = torch.stack(rows)
    if n_invalid > 0:
        # Scatter some -1 like compress causal invalids.
        flat = idx.view(-1)
        n = min(n_invalid, flat.numel())
        pos = torch.randperm(flat.numel(), device=device, generator=g)[:n]
        flat[pos] = -1
        idx = flat.view_as(idx)
    do = torch.randn_like(q)
    scale = dim**-0.5
    return q, kv, sink, idx, do, scale


def run_kernel(fn, q, kv, sink, idx, do, scale):
    q_ = q.detach().clone().requires_grad_(True)
    kv_ = kv.detach().clone().requires_grad_(True)
    sink_ = sink.detach().clone().requires_grad_(True)
    o = fn(q_, kv_, sink_, idx, scale)
    o.backward(do)
    out = {
        "o": o.detach().cpu(),
        "dq": q_.grad.detach().cpu(),
        "dkv": kv_.grad.detach().cpu(),
        "dsink": sink_.grad.detach().cpu(),
    }
    del q_, kv_, sink_, o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_ref(q, kv, sink, idx, do, scale):
    q_ = q.detach().float().clone().requires_grad_(True)
    kv_ = kv.detach().float().clone().requires_grad_(True)
    sink_ = sink.detach().float().clone().requires_grad_(True)
    o = ref_sparse_mla(q_, kv_, sink_, idx, scale)
    o.backward(do.float())
    return {
        "o": o.detach().to(q.dtype),
        "dq": q_.grad.detach().to(q.dtype),
        "dkv": kv_.grad.detach().to(kv.dtype),
        "dsink": sink_.grad.detach().float(),
    }


def compare_pair(title: str, a: dict, b: dict, a_name: str, b_name: str) -> list[Diff]:
    print(f"\n=== {title}  ({b_name} vs {a_name}; sumsq ratio is {b_name}/{a_name}) ===")
    diffs = []
    for key in ("o", "dq", "dkv", "dsink"):
        d = compute_diff(key, a[key], b[key])
        print(fmt(d))
        diffs.append(d)
    return diffs


PRESETS = {
    "tiny": dict(batch=1, seqlen=64, heads=8, dim=512, seqlen_kv=80, topk=64, n_invalid=0, with_ref=True),
    "small": dict(batch=1, seqlen=128, heads=16, dim=512, seqlen_kv=160, topk=128, n_invalid=256, with_ref=True),
    # Last-PP compress layer from 2-node dump: q=(1,2048,16,512) kv=(1,2560,512) topk=(1,2048,640)
    "flash": dict(batch=1, seqlen=2048, heads=16, dim=512, seqlen_kv=2560, topk=640, n_invalid=0, with_ref=False),
    # Window-only last-PP neighbor: kv=2064 topk=144
    "flash_win": dict(batch=1, seqlen=2048, heads=16, dim=512, seqlen_kv=2064, topk=144, n_invalid=25520, with_ref=False),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small", choices=list(PRESETS) + ["all"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ref", action="store_true", help="Force FP32 torch reference even on flash presets")
    p.add_argument(
        "--tilelang-impl",
        choices=("lumen", "miles"),
        default="lumen",
        help="TileLang autograd wrapper to test",
    )
    args = p.parse_args()

    if args.tilelang_impl == "miles":
        from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla import (
            sparse_attn_tilelang,
        )
    else:
        from lumen.kernels.dsv4.sparse_mla.tilelang_sparse_mla import sparse_attn_tilelang
    from lumen.kernels.dsv4.sparse_mla.triton_sparse_mla import sparse_attn_triton

    names = list(PRESETS) if args.preset == "all" else [args.preset]
    failed = 0
    for name in names:
        cfg = dict(PRESETS[name])
        with_ref = bool(args.ref or cfg.pop("with_ref"))
        print(f"\n######## preset={name} {cfg} seed={args.seed} ########")
        q, kv, sink, idx, do, scale = make_inputs(seed=args.seed, device=args.device, **cfg)
        print(f"q={tuple(q.shape)} kv={tuple(kv.shape)} topk={tuple(idx.shape)} n_neg={(idx < 0).sum().item()}")

        tl = run_kernel(sparse_attn_tilelang, q, kv, sink, idx, do, scale)
        tr = run_kernel(sparse_attn_triton, q, kv, sink, idx, do, scale)
        diffs = compare_pair("TileLang vs Triton", tl, tr, "TileLang", "Triton")
        dkv = next(d for d in diffs if d.name == "dkv")
        dq = next(d for d in diffs if d.name == "dq")
        print(
            f"  (train dump L43: Triton/TileLang dkv sumsq~1.63 dq~1.00; "
            f"this run dkv_sumsq={dkv.sumsq_ratio:.3f} dkv_cos={dkv.cosine:.4f} dq={dq.sumsq_ratio:.3f})"
        )
        if any(d.cosine < 0.99 or d.sumsq_ratio < 0.7 or d.sumsq_ratio > 1.4 for d in diffs):
            failed += 1

        if with_ref:
            ref = run_ref(q, kv, sink, idx, do, scale)
            compare_pair("TileLang vs FP32 ref", ref, tl, "ref", "TileLang")
            compare_pair("Triton vs FP32 ref", ref, tr, "ref", "Triton")

    if failed:
        print(f"\nRESULT: {failed} preset(s) show a TileLang/Triton backward gap")
        return 1
    print("\nRESULT: TileLang and Triton match on reported presets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
