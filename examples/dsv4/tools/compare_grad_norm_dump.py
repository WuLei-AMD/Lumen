#!/usr/bin/env python3
"""Diff per-parameter grad norms between two DSV4_DUMP_GRAD_NORM=1 dumps.

Each dump dir holds one ``w{world}_pp{pp}_tp{tp}_ep{ep}.tsv`` per rank with
columns ``gsq tp sp ar inc shape name``.  Only rows with ``inc=1`` enter the
global grad norm, so ``sqrt(sum(gsq[inc]))`` reproduces Megatron's ``grad norm``
exactly.

Usage:
    compare_grad_norm_dump.py LUMEN_DIR MILES_DIR [--top N] [--bucket B]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import defaultdict

BUCKET_RULES = [
    ("indexer", r"\.indexer\."),
    ("compressor", r"\.compressor\."),
    ("shared_expert", r"shared_expert"),
    ("expert", r"\.experts\.|\.local_experts\."),
    ("attn", r"self_attention"),
    ("norm", r"norm|layernorm"),
    ("embed_out", r"embedding|output_layer"),
]


def bucket_of(name: str) -> str:
    for label, pat in BUCKET_RULES:
        if re.search(pat, name):
            return label
    return "other"


def load(dump_dir: str) -> dict[tuple[str, str], float]:
    """Map (rank_key, param_name) -> gsq for rows included in the norm.

    Duplicated (non-TP) params are only included on tp0, so keying by rank
    keeps the sum equal to the reported grad norm.
    """
    rows: dict[tuple[str, str], float] = {}
    files = sorted(glob.glob(os.path.join(dump_dir, "w*_tp*_ep*.tsv")))
    if not files:
        raise SystemExit(f"no dump tsv found under {dump_dir}")
    for path in files:
        rank = re.sub(r"^w\d+_", "", os.path.basename(path)[: -len(".tsv")])
        with open(path) as fh:
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 7 or cols[0] == "gsq":
                    continue
                if cols[4] != "1":
                    continue
                rows[(rank, cols[6])] = float(cols[0])
    return rows


def strip_prefix(name: str) -> str:
    return re.sub(r"^(module\.)+", "", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lumen_dir")
    ap.add_argument("miles_dir")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--bucket", default=None, help="only show this bucket")
    args = ap.parse_args()

    lum = {(r, strip_prefix(n)): v for (r, n), v in load(args.lumen_dir).items()}
    mil = {(r, strip_prefix(n)): v for (r, n), v in load(args.miles_dir).items()}

    tot_l, tot_m = sum(lum.values()), sum(mil.values())
    print(f"Lumen gn = {math.sqrt(tot_l):.4f}   Miles gn = {math.sqrt(tot_m):.4f}   ratio = {math.sqrt(tot_l / tot_m):.4f}")

    bl, bm = defaultdict(float), defaultdict(float)
    for (_, name), v in lum.items():
        bl[bucket_of(name)] += v
    for (_, name), v in mil.items():
        bm[bucket_of(name)] += v

    print(f"\n{'bucket':16} {'Lumen sq':>12} {'Miles sq':>12} {'gn ratio':>9} {'sq delta':>12}")
    for b in sorted(set(bl) | set(bm), key=lambda b: -(bl[b] + bm[b])):
        ratio = math.sqrt(bl[b] / bm[b]) if bm[b] else float("inf")
        print(f"{b:16} {bl[b]:12.4f} {bm[b]:12.4f} {ratio:9.3f} {bl[b] - bm[b]:12.4f}")

    only_l = set(lum) - set(mil)
    only_m = set(mil) - set(lum)
    if only_l or only_m:
        print(f"\nname mismatch: only-Lumen={len(only_l)} only-Miles={len(only_m)}")
        for k in list(only_l)[:5]:
            print(f"  only-Lumen {k[0]} {k[1]}")
        for k in list(only_m)[:5]:
            print(f"  only-Miles {k[0]} {k[1]}")

    shared = set(lum) & set(mil)
    print(f"\ntop {args.top} params by |sq delta| (shared names, n={len(shared)})")
    print(f"{'Lumen sq':>12} {'Miles sq':>12} {'ratio':>8}  name")
    ranked = sorted(shared, key=lambda k: -abs(lum[k] - mil[k]))
    shown = 0
    for k in ranked:
        if args.bucket and bucket_of(k[1]) != args.bucket:
            continue
        lv, mv = lum[k], mil[k]
        r = math.sqrt(lv / mv) if mv else float("inf")
        print(f"{lv:12.5g} {mv:12.5g} {r:8.3f}  {k[0]} {k[1]}")
        shown += 1
        if shown >= args.top:
            break


if __name__ == "__main__":
    main()
