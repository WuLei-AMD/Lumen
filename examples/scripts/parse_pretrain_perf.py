#!/usr/bin/env python3
"""Parse Megatron iteration logs into a compact perf table.

Reads lines like:
  iteration        3/      50 | ... elapsed time per iteration (ms): 7916.7 | mem usages: 0.7700 | ...
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s+(\d+)\s+\|"
    r".*?elapsed time per iteration \(ms\):\s+([0-9.]+)"
    r".*?mem usages:\s+([0-9.]+)"
)
DONE_RE = re.compile(r"\[after training is done\]")


def parse_log(log_path: Path, gbs: int, seq_len: int, skip_warmup: int) -> dict:
    rows: dict[int, tuple[float, float]] = {}
    done = False
    if log_path.exists():
        text = log_path.read_text(errors="replace")
        done = bool(DONE_RE.search(text))
        for m in ITER_RE.finditer(text):
            it = int(m.group(1))
            rows[it] = (float(m.group(3)), float(m.group(4)))

    iters = sorted(rows)
    times = [rows[i][0] for i in iters]
    mems = [rows[i][1] for i in iters]
    steady = times[skip_warmup:] if len(times) > skip_warmup else times
    mean_ms = statistics.mean(steady) if steady else math.nan
    median_ms = statistics.median(steady) if steady else math.nan
    tok_per_iter = gbs * seq_len
    tok_s = (tok_per_iter / (mean_ms / 1000.0)) if steady else math.nan

    status = "ok" if done and iters else ("partial" if iters else "missing")
    if not log_path.exists():
        status = "missing"

    return {
        "log": str(log_path),
        "status": status,
        "n_logged": len(iters),
        "n_steady": len(steady),
        "skip_warmup": skip_warmup,
        "iter_ms_mean": round(mean_ms, 1) if steady else None,
        "iter_ms_median": round(median_ms, 1) if steady else None,
        "iter_ms_min": round(min(steady), 1) if steady else None,
        "iter_ms_max": round(max(steady), 1) if steady else None,
        "iter_ms_first": round(times[0], 1) if times else None,
        "tokens_per_sec": round(tok_s, 1) if steady else None,
        "mem_usages_last": round(mems[-1], 4) if mems else None,
        "gbs": gbs,
        "seq_len": seq_len,
        "iters": iters,
        "times_ms": [round(t, 1) for t in times],
    }


def fmt(v, width=10, nd=1):
    if v is None:
        return f"{'n/a':>{width}}"
    if isinstance(v, float):
        return f"{v:{width}.{nd}f}"
    return f"{v:>{width}}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-warmup", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out_dir)

    jobs = [
        ("llama2-7b", "bf16", out / "llama2" / "lumen_llama2_7b_bf16.log", 256, 4096),
        ("llama2-7b", "fp8", out / "llama2" / "lumen_llama2_7b_fp8.log", 256, 4096),
        ("llama3-8b", "bf16", out / "llama31" / "lumen_llama31_8b_bf16.log", 128, 8192),
        ("llama3-8b", "fp8", out / "llama31" / "lumen_llama31_8b_fp8.log", 128, 8192),
        ("qwen3-8b", "bf16", out / "qwen3" / "lumen_qwen3_8b_bf16.log", 128, 8192),
        ("qwen3-8b", "fp8", out / "qwen3" / "lumen_qwen3_8b_fp8.log", 128, 8192),
    ]

    results = []
    for model, prec, log, gbs, seq in jobs:
        rec = parse_log(log, gbs, seq, args.skip_warmup)
        rec["model"] = model
        rec["precision"] = prec
        results.append(rec)

    summary_txt = out / "SUMMARY.txt"
    summary_json = out / "SUMMARY.json"
    summary_csv = out / "SUMMARY.csv"

    lines = [
        "OPUS-attn dense pretrain perf  (skip first "
        f"{args.skip_warmup} iter(s) as warmup)",
        f"tokens/s = GBS * SEQ_LEN / mean_iter_s",
        "",
        f"{'model':<12} {'prec':<6} {'status':<8} {'n':>4} "
        f"{'mean_ms':>10} {'med_ms':>10} {'min_ms':>10} {'max_ms':>10} "
        f"{'tok/s':>12} {'mem':>8}  log",
        "-" * 120,
    ]
    csv = [
        "model,precision,status,n_logged,n_steady,iter_ms_mean,iter_ms_median,"
        "iter_ms_min,iter_ms_max,iter_ms_first,tokens_per_sec,mem_usages_last,gbs,seq_len,log"
    ]
    for r in results:
        log_rel = r["log"]
        lines.append(
            f"{r['model']:<12} {r['precision']:<6} {r['status']:<8} {r['n_logged']:>4} "
            f"{fmt(r['iter_ms_mean'])} {fmt(r['iter_ms_median'])} "
            f"{fmt(r['iter_ms_min'])} {fmt(r['iter_ms_max'])} "
            f"{fmt(r['tokens_per_sec'], 12, 1)} {fmt(r['mem_usages_last'], 8, 4)}  {log_rel}"
        )
        csv.append(
            ",".join(
                str(x) if x is not None else ""
                for x in [
                    r["model"],
                    r["precision"],
                    r["status"],
                    r["n_logged"],
                    r["n_steady"],
                    r["iter_ms_mean"],
                    r["iter_ms_median"],
                    r["iter_ms_min"],
                    r["iter_ms_max"],
                    r["iter_ms_first"],
                    r["tokens_per_sec"],
                    r["mem_usages_last"],
                    r["gbs"],
                    r["seq_len"],
                    r["log"],
                ]
            )
        )

    summary_txt.write_text("\n".join(lines) + "\n")
    summary_csv.write_text("\n".join(csv) + "\n")
    slim = [{k: v for k, v in r.items() if k not in ("iters", "times_ms")} for r in results]
    summary_json.write_text(json.dumps({"skip_warmup": args.skip_warmup, "jobs": slim}, indent=2) + "\n")
    print(summary_txt.read_text())


if __name__ == "__main__":
    main()
