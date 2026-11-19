#!/usr/bin/env python3
"""Run and summarize the Qwen3-30B-A3B GBS=256 benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
MBS = int(os.environ.get("MBS", "2"))
GBS = int(os.environ.get("GBS", "256"))
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "20"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "4096"))

BASE_ENV = {
    "MODEL_PATH": os.environ.get("MODEL_PATH", "/nobackup/model/Qwen3-30B-A3B"),
    "DATA_PATH": os.environ.get(
        "DATA_PATH", "/nobackup/data/fineweb-sample-10BT-26624.jsonl"
    ),
    "MEGATRON_LOAD_PATH": os.environ.get(
        "MEGATRON_LOAD_PATH",
        "/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8",
    ),
    "TRAIN_STEPS": str(TRAIN_STEPS),
    "SEQ_LEN": str(SEQ_LEN),
    "MBS": str(MBS),
    "GBS": str(GBS),
}

CONTAINER_RESULTS = "/workspace/Lumen/examples/qwen3-30b-a3b/results"
CASE_CONFIGS: dict[str, dict[str, str]] = {
    "fsdp_sequential": {
        "EXPERT_BACKEND": "sequential",
        "RUN_SUFFIX": "real-optimized20",
        "COMMAND": "bash run_fsdp.sh",
    },
    "fsdp_sonic_multistream": {
        "EXPERT_BACKEND": "sonic",
        "SONIC_MOE_GEMM_BACKEND": "triton",
        "SONIC_MOE_GROUPED_GEMM_BACKEND": "multistream",
        "SONIC_MOE_MULTISTREAM_PRIORITY": "0",
        "MOE_GLOBAL_EXPERT_LAYOUT": "1",
        "RUN_SUFFIX": "real-global-layout-priority0-optimized20",
        "COMMAND": "bash run_fsdp.sh",
    },
    "fsdp_te_grouped": {
        "EXPERT_BACKEND": "te_grouped",
        "NVTE_USE_CUTLASS_GROUPED_GEMM": "1",
        "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK": "1",
        "RUN_SUFFIX": "real-ck-optimized20",
        "COMMAND": "bash run_fsdp.sh",
    },
    "sequential": {
        "MOE_IMPL": "sequential",
        "RUN_SUFFIX": "real-optimized20",
    },
    "sonic_triton_tune": {
        "COMMAND": (
            "python tune_sonic_grouped_gemm.py --tokens 131072 "
            "--output results/sonic_grouped_gemm_qwen3_gbs256_tuned.json"
        )
    },
    "sonic_triton": {
        "MOE_IMPL": "sonic",
        "SONIC_MOE_GEMM_BACKEND": "triton",
        "RUN_SUFFIX": "triton-tuned",
    },
    "sonic_multistream": {
        "MOE_IMPL": "sonic",
        "SONIC_MOE_GEMM_BACKEND": "triton",
        "SONIC_MOE_GROUPED_GEMM_BACKEND": "multistream",
        "SONIC_MOE_MULTISTREAM_PRIORITY": "0",
        "RUN_SUFFIX": "real-multistream-priority0-optimized20",
    },
    "te_grouped": {"MOE_IMPL": "te_grouped"},
    "te_ck": {
        "MOE_IMPL": "te_grouped",
        "NVTE_USE_CUTLASS_GROUPED_GEMM": "1",
        "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK": "1",
        "RUN_SUFFIX": "real-ck-optimized20",
    },
    "te_hipblaslt_tune": {
        "MOE_IMPL": "te_grouped",
        "TE_HIPBLASLT_ALGO_SAVE": (
            f"{CONTAINER_RESULTS}/qwen3-30b-a3b-te-hipblaslt-gbs256.cache"
        ),
        "TE_HIPBLASLT_TUNING_RUN_COUNT": "5",
        "TE_HIPBLASLT_TUNING_ALGO_COUNT": "64",
        "TRAIN_STEPS": "1",
        "LR_WARMUP_ITERS": "0",
        "RUN_SUFFIX": "real-hipblaslt-tuning",
    },
    "te_hipblaslt_tuned": {
        "MOE_IMPL": "te_grouped",
        "TE_HIPBLASLT_ALGO_LOAD": (
            f"{CONTAINER_RESULTS}/qwen3-30b-a3b-te-hipblaslt-gbs256.cache"
        ),
        "RUN_SUFFIX": "real-hipblaslt",
    },
}

MEGATRON_CASES = (
    ("sequential", 29900),
    ("sonic_multistream", 29901),
    ("te_ck", 29906),
)
FSDP_CASES = (
    ("FSDP Sequential", "fsdp_sequential", 29920),
    ("FSDP Sonic multi-stream BLAS", "fsdp_sonic_multistream", 29921),
    ("FSDP TE Grouped", "fsdp_te_grouped", 29922),
)

MEGATRON_PATTERN = re.compile(
    r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
    r".*?global batch size:\s*(\d+).*?lm loss:\s*([0-9.Ee+-]+)"
)
FSDP_PATTERN = re.compile(
    r"step\s+(\d+)/\d+\s+\|\s+loss\s+[0-9.Ee+-]+\s+\|\s+"
    r"lm_loss\s+([0-9.Ee+-]+).*?step_time_ms\s+([0-9.]+).*?"
    r"throughput_samples_per_sec\s+([0-9.]+).*?peak_memory_mb\s+([0-9.]+)"
)
MEGATRON_MEMORY_PATTERN = re.compile(r"max allocated:\s*([0-9.]+)")
MBS_PATTERN = re.compile(r"-mbs(\d+)")
GBS_PATTERN = re.compile(r"-gbs(\d+)")


def run_case(case_name: str, port: int) -> bool:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(CASE_CONFIGS[case_name])
    env["MASTER_PORT"] = str(port)
    print(f"START_CASE={case_name}", flush=True)
    result = subprocess.run(["bash", str(SCRIPT_DIR / "run_docker.sh")], env=env)
    status = "DONE_CASE" if result.returncode == 0 else "FAILED_CASE"
    print(f"{status}={case_name}", flush=True)
    return result.returncode == 0


def run_matrix(mode: str) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    if mode == "megatron":
        for case_name, port in MEGATRON_CASES:
            failed |= not run_case(case_name, port)
    else:
        status_path = RESULTS_DIR / "qwen3-30b-a3b-fsdp-real-status.csv"
        with status_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("case", "status", "mbs"))
            for label, case_name, port in FSDP_CASES:
                completed = run_case(case_name, port)
                writer.writerow((label, "COMPLETED" if completed else "FAILED", MBS))
                handle.flush()
                failed |= not completed
    return int(failed)


def parse_log(path: Path) -> tuple[list[dict[str, float]], int, int, float]:
    text = path.read_text(errors="replace")
    mbs_match = MBS_PATTERN.search(path.name)
    gbs_match = GBS_PATTERN.search(path.name)
    mbs = int(mbs_match.group(1)) if mbs_match else MBS
    gbs = int(gbs_match.group(1)) if gbs_match else GBS
    points = [
        {
            "step": int(step),
            "time_ms": float(time_ms),
            "loss": float(loss),
            "throughput": int(logged_gbs) * 1000 / float(time_ms),
        }
        for step, time_ms, logged_gbs, loss in MEGATRON_PATTERN.findall(text)
        if int(logged_gbs) == gbs
    ]
    # Megatron intentionally logs iteration 1 without resetting its interval
    # accumulators. With log_interval=1, iteration 2 is therefore the average
    # of iterations 1 and 2 rather than a per-step value. Recover the actual
    # second point for both the loss and timing curves.
    if len(points) >= 2 and points[0]["step"] == 1 and points[1]["step"] == 2:
        points[1]["loss"] = 2.0 * points[1]["loss"] - points[0]["loss"]
        points[1]["time_ms"] = 2.0 * points[1]["time_ms"] - points[0]["time_ms"]
        points[1]["throughput"] = gbs * 1000 / points[1]["time_ms"]
    if not points:
        matches = FSDP_PATTERN.findall(text)
        points = [
            {
                "step": int(step),
                "time_ms": float(time_ms),
                "loss": float(loss),
                "throughput": float(throughput),
                "memory_mb": float(memory_mb),
            }
            for step, loss, time_ms, throughput, memory_mb in matches
        ]
    memory_values = [
        float(value) for value in MEGATRON_MEMORY_PATTERN.findall(text)
    ]
    memory_values.extend(
        point["memory_mb"] for point in points if "memory_mb" in point
    )
    return points, mbs, gbs, max(memory_values, default=float("nan"))


def case_label(path: Path) -> str:
    labels = (
        ("fsdp-sequential-real-optimized20-", "FSDP Sequential"),
        ("fsdp-sonic-real-global-layout-priority0-", "FSDP Sonic"),
        ("fsdp-te_grouped-real-ck-", "FSDP TE CK"),
        ("sequential-real-optimized20-", "Megatron Sequential"),
        ("sonic-real-multistream-priority0-", "Megatron Sonic"),
        ("te_grouped-real-hipblaslt-", "Megatron TE hipBLASLt"),
        ("te_grouped-real-ck-", "Megatron TE CK"),
    )
    for marker, label in labels:
        if marker in path.stem:
            return label
    return path.stem.removeprefix("qwen3-30b-a3b-")


def expected_logs() -> list[Path]:
    names = (
        f"qwen3-30b-a3b-sequential-real-optimized20-seq4096-mbs{MBS}-gbs{GBS}.log",
        f"qwen3-30b-a3b-sonic-real-multistream-priority0-optimized20-seq4096-mbs{MBS}-gbs{GBS}.log",
        f"qwen3-30b-a3b-te_grouped-real-ck-optimized20-seq4096-mbs{MBS}-gbs{GBS}.log",
        f"qwen3-30b-a3b-fsdp-sequential-real-optimized20-bf16-seq4096-mbs{MBS}-gbs{GBS}.log",
        f"qwen3-30b-a3b-fsdp-sonic-real-global-layout-priority0-optimized20-bf16-seq4096-mbs{MBS}-gbs{GBS}.log",
        f"qwen3-30b-a3b-fsdp-te_grouped-real-ck-optimized20-bf16-seq4096-mbs{MBS}-gbs{GBS}.log",
    )
    return [RESULTS_DIR / name for name in names]


def summarize(warmup_steps: int) -> None:
    logs = [path for path in expected_logs() if path.exists()]
    runs = []
    for path in logs:
        points, mbs, gbs, peak_memory_mb = parse_log(path)
        if not points:
            continue
        steady = [
            point["time_ms"] for point in points if point["step"] > warmup_steps
        ] or [point["time_ms"] for point in points]
        runs.append((path, points, mbs, gbs, steady, peak_memory_mb))
    if not runs:
        raise SystemExit("No completed GBS=256 training steps found")

    summary_path = RESULTS_DIR / "qwen3-30b-a3b-te-vs-fsdp-summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "status",
                "steps",
                "mbs",
                "gbs",
                "median_step_ms",
                "mean_step_ms",
                "samples_per_second",
                "peak_memory_gib",
                "final_loss",
            ],
        )
        writer.writeheader()
        for path, points, mbs, gbs, steady, peak_memory_mb in runs:
            median_ms = statistics.median(steady)
            writer.writerow(
                {
                    "case": case_label(path),
                    "status": "COMPLETED",
                    "steps": len(points),
                    "mbs": mbs,
                    "gbs": gbs,
                    "median_step_ms": f"{median_ms:.1f}",
                    "mean_step_ms": f"{statistics.mean(steady):.1f}",
                    "samples_per_second": f"{gbs * 1000 / median_ms:.3f}",
                    "peak_memory_gib": f"{peak_memory_mb / 1024:.3f}",
                    "final_loss": f"{points[-1]['loss']:.6g}",
                }
            )

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 7))
    for path, points, mbs, gbs, _steady, _peak_memory_mb in runs:
        axis.plot(
            [point["step"] * gbs for point in points],
            [point["loss"] for point in points],
            label=f"{case_label(path)} (MBS{mbs})",
            linewidth=1.8,
        )
    axis.set_title(f"Qwen3-30B-A3B BF16 LM loss, GBS={GBS}")
    axis.set_xlabel("Consumed samples")
    axis.set_ylabel("Language-model loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    loss_path = RESULTS_DIR / "qwen3-30b-a3b-te-vs-fsdp-lm-loss.png"
    figure.savefig(loss_path, dpi=160)
    plt.close(figure)

    labels = [
        f"{case_label(path)} (MBS{mbs})"
        for path, _points, mbs, _gbs, _steady, _memory in runs
    ]
    rates = [
        gbs * 1000 / statistics.median(steady)
        for _path, _points, _mbs, gbs, steady, _memory in runs
    ]
    elapsed = [
        statistics.median(steady) / 1000
        for _path, _points, _mbs, _gbs, steady, _memory in runs
    ]
    memory = [
        peak_memory_mb / 1024
        for _path, _points, _mbs, _gbs, _steady, peak_memory_mb in runs
    ]
    order = sorted(range(len(rates)), key=rates.__getitem__)
    figure, axes = plt.subplots(1, 3, figsize=(22, 8))
    ordered_labels = [labels[index] for index in order]
    metrics = (
        ("Peak allocated memory", "GiB", memory, "%.1f"),
        ("Throughput", "Samples / second", rates, "%.2f"),
        ("Elapsed time per iteration", "Seconds", elapsed, "%.2f"),
    )
    for axis, (title, xlabel, values, value_format) in zip(
        axes, metrics, strict=True
    ):
        bars = axis.barh(ordered_labels, [values[index] for index in order])
        axis.bar_label(bars, fmt=value_format, padding=4)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.25)
    max_step = max(point["step"] for _path, points, *_rest in runs for point in points)
    figure.suptitle(
        f"Qwen3-30B-A3B steady-state performance "
        f"(GBS={GBS}, steps {warmup_steps + 1}-{max_step})"
    )
    figure.tight_layout()
    performance_path = RESULTS_DIR / "qwen3-30b-a3b-te-vs-fsdp-performance.png"
    figure.savefig(performance_path, dpi=160)
    plt.close(figure)
    print(summary_path)
    print(performance_path)
    print(loss_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("megatron", help="run the Megatron matrix")
    subparsers.add_parser("fsdp", help="run the FSDP matrix")
    case_parser = subparsers.add_parser("case", help="run one named case")
    case_parser.add_argument("name", choices=sorted(CASE_CONFIGS))
    case_parser.add_argument("--port", type=int, default=29950)
    summary_parser = subparsers.add_parser("summarize", help="create CSV and plots")
    summary_parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    if args.command in {"megatron", "fsdp"}:
        return run_matrix(args.command)
    if args.command == "case":
        return int(not run_case(args.name, args.port))
    summarize(args.warmup_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
