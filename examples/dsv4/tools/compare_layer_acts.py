#!/usr/bin/env python3
"""Compare Lumen vs Miles per-layer hidden dumps from DSV4_DUMP_LAYER_ACT.

Usage:
  python examples/dsv4/tools/compare_layer_acts.py \\
    --lumen-dir /data/leiwu/models/dsv4_layer_act_lumen \\
    --lumen-dir /tmp/dsv4_layer_act_lumen_worker \\
    --miles-dir /data/leiwu/models/dsv4_layer_act_miles \\
    --miles-dir /tmp/dsv4_layer_act_miles_worker

PP ranks dump locally. Collect worker MODEL_DIR dumps onto the head
before comparing (repeat --lumen-dir / --miles-dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load_by_stage(dump_dirs: list[Path], tag: str, layer: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for dump_dir in dump_dirs:
        if not dump_dir.is_dir():
            raise SystemExit(f"dump dir missing: {dump_dir}")
        for path in sorted(dump_dir.glob(f"{tag}_pp*_l{layer:02d}_*_fwd0.pt")):
            obj = torch.load(path, map_location="cpu", weights_only=False)
            stage = str(obj.get("stage") or path.name)
            if stage == "layer_out":
                continue
            hidden = obj["hidden"].float()
            if stage in out:
                raise SystemExit(f"duplicate stage {stage} ({path})")
            out[stage] = hidden
    return out


def _load_by_layer(dump_dirs: list[Path], tag: str) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for dump_dir in dump_dirs:
        if not dump_dir.is_dir():
            raise SystemExit(f"dump dir missing: {dump_dir}")
        for path in sorted(dump_dir.glob(f"{tag}_pp*_l*_fwd0.pt")):
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if obj.get("stage") not in (None, "layer_out"):
                continue
            layer = int(obj["layer"])
            hidden = obj["hidden"].float().reshape(-1)
            if layer in out:
                raise SystemExit(f"duplicate layer {layer} ({path})")
            out[layer] = hidden
    return out


def _as_token_expert(t: torch.Tensor, n_experts: int | None = None) -> torch.Tensor:
    if t.dim() >= 2 and t.shape[-1] <= 1024:
        return t.reshape(-1, t.shape[-1])
    n_exp = n_experts or 256
    if t.numel() % n_exp == 0:
        return t.reshape(-1, n_exp)
    return t.reshape(t.shape[0], -1)


def _print_router_stats(map_a: torch.Tensor, map_b: torch.Tensor, probs_a, probs_b) -> None:
    a = _as_token_expert(map_a) > 0.5
    b = _as_token_expert(map_b, n_experts=a.shape[-1]) > 0.5
    n = min(a.shape[0], b.shape[0])
    a = a[:n]
    b = b[:n]
    same = (a == b).all(dim=-1)
    flipped = int((~same).sum())
    hamming = int((a != b).sum())
    overlap = (a & b).sum(dim=-1).float()
    disjoint = int((overlap == 0).sum())
    k = float(a.sum(dim=-1).float().mean())
    print(
        f"router_map tokens={n} experts={a.shape[-1]} exact_match={int(same.sum())}/{n} "
        f"flipped_tokens={flipped} hamming={hamming} mean_overlap={float(overlap.mean()):.3f}/{k:.1f} "
        f"fully_disjoint={disjoint}/{n}"
    )
    if flipped:
        idx = (~same).nonzero(as_tuple=False).flatten()[:8]
        for i in idx.tolist():
            ea = a[i].nonzero(as_tuple=False).flatten().tolist()
            eb = b[i].nonzero(as_tuple=False).flatten().tolist()
            print(f"  token {i}: lumen={ea} miles={eb}")
    if probs_a is not None and probs_b is not None:
        pa = _as_token_expert(probs_a, n_experts=a.shape[-1])[:n]
        pb = _as_token_expert(probs_b, n_experts=a.shape[-1])[:n]
        diff = (pa - pb).abs()
        selected = a | b
        sel_mean = float(diff[selected].mean()) if int(selected.sum()) else 0.0
        print(
            f"router_probs max_abs={float(diff.max()):.4e} mean_abs={float(diff.mean()):.4e} "
            f"on_selected mean_abs={sel_mean:.4e}"
        )


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.reshape(-1)
    b = b.reshape(-1)
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]
    diff = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return {
        "n": float(n),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rms": float((diff.square().mean()).sqrt()),
        "cosine": float(cos),
        "len_match": float(a.numel() == b.numel()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lumen-dir", type=Path, action="append", required=True)
    p.add_argument("--miles-dir", type=Path, action="append", required=True)
    p.add_argument("--lumen-tag", default="lumen")
    p.add_argument("--miles-tag", default="miles")
    p.add_argument(
        "--inner-layer",
        type=int,
        action="append",
        default=None,
        help="compare layer-N inner stages (repeat to compare several layers)",
    )
    args = p.parse_args()

    if args.inner_layer:
        for layer in args.inner_layer:
            print(f"\n===== inner layer {layer} =====")
            lumen = _load_by_stage(args.lumen_dir, args.lumen_tag, layer)
            miles = _load_by_stage(args.miles_dir, args.miles_tag, layer)
            stages = sorted(set(lumen) & set(miles))
            missing_l = sorted(set(miles) - set(lumen))
            missing_m = sorted(set(lumen) - set(miles))
            if missing_l:
                print(f"missing in lumen dumps: {missing_l}")
            if missing_m:
                print(f"missing in miles dumps: {missing_m}")
            if not stages:
                print("no overlapping inner stages")
                continue
            print(
                f"{'stage':>22}  {'max_abs':>12}  {'mean_abs':>12}  {'rms':>12}  {'cosine':>9}  {'n':>10}"
            )
            first_bad = None
            for stage in stages:
                st = _stats(lumen[stage], miles[stage])
                print(
                    f"{stage:>22}  {st['max_abs']:12.4e}  {st['mean_abs']:12.4e}  "
                    f"{st['rms']:12.4e}  {st['cosine']:9.6f}  {int(st['n']):10d}"
                )
                if first_bad is None and st["cosine"] < 0.999:
                    first_bad = stage
            if first_bad is None:
                print("all overlapping stages cosine >= 0.999")
            else:
                print(f"first stage with cosine < 0.999: {first_bad}")
            if "router_map" in lumen and "router_map" in miles:
                _print_router_stats(
                    lumen["router_map"],
                    miles["router_map"],
                    lumen.get("router_probs"),
                    miles.get("router_probs"),
                )
        return

    lumen = _load_by_layer(args.lumen_dir, args.lumen_tag)
    miles = _load_by_layer(args.miles_dir, args.miles_tag)
    layers = sorted(set(lumen) & set(miles))
    missing_l = sorted(set(miles) - set(lumen))
    missing_m = sorted(set(lumen) - set(miles))
    if missing_l:
        print(f"missing in lumen dumps: {missing_l}")
    if missing_m:
        print(f"missing in miles dumps: {missing_m}")
    if not layers:
        raise SystemExit("no overlapping layers")

    print(
        f"{'layer':>6}  {'max_abs':>12}  {'mean_abs':>12}  {'rms':>12}  {'cosine':>9}  {'n':>10}"
    )
    first_bad = None
    for layer in layers:
        st = _stats(lumen[layer], miles[layer])
        print(
            f"{layer:6d}  {st['max_abs']:12.4e}  {st['mean_abs']:12.4e}  "
            f"{st['rms']:12.4e}  {st['cosine']:9.6f}  {int(st['n']):10d}"
        )
        if first_bad is None and st["cosine"] < 0.999:
            first_bad = layer
    if first_bad is None:
        print("all overlapping layers cosine >= 0.999")
    else:
        print(f"first layer with cosine < 0.999: {first_bad}")


if __name__ == "__main__":
    main()
