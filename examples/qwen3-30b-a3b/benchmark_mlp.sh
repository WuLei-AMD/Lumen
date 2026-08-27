#!/usr/bin/env bash
# End-to-end SequentialMLP versus TEGroupedMLP benchmark.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR=${RESULTS_DIR:-"${SCRIPT_DIR}/results"}
TRAIN_STEPS=${TRAIN_STEPS:-20}

for implementation in sequential te_grouped; do
    MOE_IMPL="${implementation}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
        bash "${SCRIPT_DIR}/run_pretrain.sh"
done

python3 - "${RESULTS_DIR}" <<'PY'
import csv
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
pattern = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
rows = []
for name in ("sequential", "te_grouped"):
    logs = sorted(root.glob(f"qwen3-30b-a3b-{name}-*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise SystemExit(f"No log found for {name}")
    timings = [float(value) for value in pattern.findall(logs[-1].read_text())][2:]
    if not timings:
        raise SystemExit(f"No steady-state timings found in {logs[-1]}")
    rows.append(
        {
            "implementation": name,
            "mean_ms": statistics.mean(timings),
            "median_ms": statistics.median(timings),
            "samples": len(timings),
        }
    )

baseline = rows[0]["mean_ms"]
for row in rows:
    row["speedup_vs_sequential"] = baseline / row["mean_ms"]

output = root / "mlp_summary.csv"
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(output.read_text())
PY
