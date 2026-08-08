#!/usr/bin/env python3
"""Merge gfx942 GEMM tune results into dsv4_bf16_tuned_gemm_mi308x.csv."""

import argparse
import csv
from pathlib import Path

LUMEN = Path(__file__).resolve().parents[2]
BASELINE = LUMEN / "examples/dsv4/configs/dsv4_bf16_tuned_gemm_mi308x.csv"
DEFAULT_OVERLAY = LUMEN / "examples/dsv4/.gemm_tune/dsv4_bf16_tuned_gfx942_mi308x.csv"

TAG_KEYS = [
    "gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle",
]
TARGET_GFX = "gfx942"
TARGET_CU = 80


def _tag(row):
    return tuple(row[k] for k in TAG_KEYS)


def _read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_rows(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV to {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def merge(baseline, overlay, output):
    base_rows = [
        r for r in _read_rows(baseline)
        if r.get("gfx") == TARGET_GFX
        and str(r.get("cu_num")) == str(TARGET_CU)
        and r.get("libtype") != "torch"
    ]
    overlay_rows = [
        r for r in _read_rows(overlay)
        if r.get("gfx") == TARGET_GFX
        and str(r.get("cu_num")) == str(TARGET_CU)
        and r.get("libtype") != "torch"
    ]

    out_map = {_tag(r): r for r in base_rows}
    replaced = 0
    added = 0
    for r in overlay_rows:
        t = _tag(r)
        if t in out_map:
            replaced += 1
        else:
            added += 1
        out_map[t] = r

    merged = sorted(
        out_map.values(),
        key=lambda r: (int(r["M"]), int(r["N"]), int(r["K"])),
    )
    _write_rows(output, merged)
    return len(merged), added + replaced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output", type=Path, default=BASELINE)
    args = parser.parse_args()

    total, n_overlay = merge(args.baseline, args.overlay, args.output)
    print(
        f"wrote {args.output}: {total} rows "
        f"(applied {n_overlay} overlay rows, torch stripped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
