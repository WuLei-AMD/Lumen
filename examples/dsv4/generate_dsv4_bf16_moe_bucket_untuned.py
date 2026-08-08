#!/usr/bin/env python3
"""Generate MoE BF16 GEMM untuned shapes using aiter get_padded_m bucketing."""

import argparse
import csv
from pathlib import Path

LUMEN = Path(__file__).resolve().parents[2]
GEMM_TUNE_DIR = LUMEN / "examples/dsv4/.gemm_tune"
DEFAULT_OUT = GEMM_TUNE_DIR / "dsv4_bf16_moe_bucket_untuned.csv"
DEFAULT_EXISTING = LUMEN / "examples/dsv4/configs/dsv4_bf16_tuned_gemm_mi308x.csv"

MOE_PAIRS = ((4096, 2048), (4096, 4096))
# 2-node flash (TP4/PP4/EP4, seq=4096): lm_head + attention/indexer + 7168-K sparse paths.
FLASH2NODE_NK_PAIRS = (
    (32320, 4096),  # lm_head / output layer (vocab 129280 / TP4)
    (8192, 1024),   # wq_b, indexer linear_wq_b (64*128 / TP4=8192)
    (4096, 256),    # wo_b row-parallel static
    (4096, 1024),   # wo_a column-parallel static
    (384, 7168),
    (512, 7168),
    (1024, 7168),
    (2048, 7168),
)
STATIC_SHAPES = (
    (4096, 1024, 4096),
    (4096, 4096, 1024),
    (4096, 512, 4096),
    (4096, 4096, 256),
    (4096, 8192, 1024),
    (4096, 64, 4096),
)
HEADER = ["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
ROW_DEFAULTS = {
    "bias": "False",
    "dtype": "torch.bfloat16",
    "outdtype": "torch.bfloat16",
    "scaleAB": "False",
    "bpreshuffle": "False",
}

# get_padded_m representatives for M in [1, 1024] on gfx942 (N=4096,K=4096).
REPRESENTATIVE_MS_SMALL = (
    1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
    240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672,
    704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
)
# get_padded_m representatives for M in [1025, 4096] on gfx942.
REPRESENTATIVE_MS_LARGE = (
    1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856,
    1920, 1984, 2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624, 2688,
    2752, 2816, 2880, 2944, 3008, 3072, 3136, 3200, 3264, 3328, 3392, 3456, 3520,
    3584, 3648, 3712, 3776, 3840, 3904, 3968, 4032, 4096,
)


def _parse_pairs(text):
    if not text:
        return MOE_PAIRS
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) % 2:
        raise ValueError("--pairs requires even count: N1,K1,N2,K2,...")
    out = []
    for i in range(0, len(parts), 2):
        out.append((int(parts[i]), int(parts[i + 1])))
    return tuple(out)


def build_shapes(pairs, min_m=1, max_m=1024, include_static=False):
    if min_m <= 1024:
        ms_small = [m for m in REPRESENTATIVE_MS_SMALL if min_m <= m <= min(max_m, 1024)]
    else:
        ms_small = []
    if max_m > 1024:
        lo = max(min_m, 1025)
        ms_large = [m for m in REPRESENTATIVE_MS_LARGE if lo <= m <= max_m]
    else:
        ms_large = []
    reps = sorted(set(ms_small + ms_large))
    shapes = [(m, n, k) for n, k in pairs for m in reps]
    if include_static:
        shapes.extend(STATIC_SHAPES)
    return sorted(set(shapes))


def _load_existing(path):
    if not path.is_file():
        return set()
    with path.open() as f:
        rows = csv.DictReader(f)
        return {
            (int(r["M"]), int(r["N"]), int(r["K"]))
            for r in rows
            if r.get("gfx", "gfx942") == "gfx942"
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument(
        "--profile",
        choices=("moe", "flash2node"),
        default="moe",
        help="moe: MoE bucket pairs; flash2node: 2-node full-model gaps (lm_head, 7168-K, etc.)",
    )
    parser.add_argument(
        "--pairs",
        default="",
        help="comma-separated N,K pairs only (e.g. 4096,2048). Overrides --profile pairs",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="skip shapes already present in --existing CSV",
    )
    parser.add_argument("--min-m", type=int, default=None)
    parser.add_argument("--max-m", type=int, default=None)
    parser.add_argument(
        "--include-static",
        action="store_true",
        help="append finetune static miss shapes (4096x1024x4096, etc.)",
    )
    args = parser.parse_args()

    if args.pairs:
        pairs = _parse_pairs(args.pairs)
    elif args.profile == "flash2node":
        pairs = FLASH2NODE_NK_PAIRS
    else:
        pairs = MOE_PAIRS

    min_m = 1 if args.min_m is None else args.min_m
    max_m = 4096 if args.max_m is None else args.max_m
    if args.profile == "moe" and args.min_m is None and args.max_m is None:
        min_m, max_m = 1, 1024

    shapes = build_shapes(
        pairs,
        min_m=min_m,
        max_m=max_m,
        include_static=args.include_static,
    )
    if args.include_existing:
        have = _load_existing(args.existing)
        shapes = [s for s in shapes if s not in have]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for m, n, k in shapes:
            row = {"M": m, "N": n, "K": k, **ROW_DEFAULTS}
            w.writerow(row)

    print(f"wrote {args.output} ({len(shapes)} shapes)")
    for n, k in pairs:
        cnt = sum(1 for m, nn, kk in shapes if (nn, kk) == (n, k))
        print(f"  N={n} K={k}: {cnt} bucket M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
