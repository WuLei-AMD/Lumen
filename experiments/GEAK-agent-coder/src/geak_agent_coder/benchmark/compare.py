"""Paired model comparison, deterministic bootstrap CIs, and reports."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import aggregate_metrics
from .models import FixedBenchmarkIdentity, TurnRecord


DEFAULT_COMPARISON_METRICS = (
    "pass_rate",
    "first_pass_rate",
    "fast_at_p.1.2",
    "turns_to_pass",
    "tokens_cost_of_pass",
    "time_cost_of_pass_s",
    "compile_error_rate",
    "correctness_error_rate",
    "monotonic_improvement_rate",
    "speedup_auc",
)


def assert_comparable_campaigns(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> FixedBenchmarkIdentity:
    """Require all fixed agent/suite/decode identities to match exactly."""

    left = FixedBenchmarkIdentity.from_mapping(baseline["fixed_identity"])
    right = FixedBenchmarkIdentity.from_mapping(candidate["fixed_identity"])
    if left != right:
        differing = [
            key
            for key in left.to_dict()
            if left.to_dict()[key] != right.to_dict()[key]
        ]
        raise ValueError(
            "campaigns are not comparable; fixed hashes differ: "
            + ", ".join(differing)
        )
    return left


def compare_variants(
    baseline_records: Iterable[TurnRecord],
    candidate_records: Iterable[TurnRecord],
    *,
    fixed_identity: FixedBenchmarkIdentity,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260921,
    confidence: float = 0.95,
    metrics: Sequence[str] = DEFAULT_COMPARISON_METRICS,
) -> dict[str, Any]:
    baseline = list(baseline_records)
    candidate = list(candidate_records)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    baseline_tasks = {row.task_id for row in baseline}
    candidate_tasks = {row.task_id for row in candidate}
    if baseline_tasks != candidate_tasks:
        raise ValueError(
            "paired comparison requires identical task IDs; "
            f"baseline_only={sorted(baseline_tasks - candidate_tasks)}, "
            f"candidate_only={sorted(candidate_tasks - baseline_tasks)}"
        )
    baseline_pairs = {(row.task_id, row.seed) for row in baseline}
    candidate_pairs = {(row.task_id, row.seed) for row in candidate}
    if baseline_pairs != candidate_pairs:
        raise ValueError(
            "paired comparison requires identical task/seed pairs"
        )
    tasks = sorted(baseline_tasks)
    baseline_metrics = aggregate_metrics(baseline)
    candidate_metrics = aggregate_metrics(candidate)
    rng = random.Random(bootstrap_seed)
    draws: dict[str, list[float]] = {metric: [] for metric in metrics}

    by_baseline = _by_task(baseline)
    by_candidate = _by_task(candidate)
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(tasks) for _ in tasks]
        left = _resample(by_baseline, sampled)
        right = _resample(by_candidate, sampled)
        left_metrics = aggregate_metrics(left, include_breakdowns=False)
        right_metrics = aggregate_metrics(right, include_breakdowns=False)
        for metric in metrics:
            left_value = _metric_value(left_metrics, metric)
            right_value = _metric_value(right_metrics, metric)
            if left_value is not None and right_value is not None:
                draws[metric].append(right_value - left_value)

    alpha = (1.0 - confidence) / 2.0
    deltas: dict[str, Any] = {}
    for metric in metrics:
        left_value = _metric_value(baseline_metrics, metric)
        right_value = _metric_value(candidate_metrics, metric)
        values = sorted(draws[metric])
        deltas[metric] = {
            "baseline": left_value,
            "candidate": right_value,
            "delta": (
                right_value - left_value
                if left_value is not None and right_value is not None
                else None
            ),
            "ci_low": _quantile(values, alpha),
            "ci_high": _quantile(values, 1.0 - alpha),
        }
    return {
        "schema_version": "geak_benchmark_comparison_v1",
        "fixed_identity": fixed_identity.to_dict(),
        "baseline_variant": baseline_metrics["variant"],
        "candidate_variant": candidate_metrics["variant"],
        "paired_tasks": len(tasks),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "unit": "task",
        },
        "metrics": deltas,
        "baseline_summary": baseline_metrics,
        "candidate_summary": candidate_metrics,
    }


def write_comparison_report(
    report: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "comparison.json"
    markdown_path = destination / "comparison.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# GEAK agent-loop comparison",
        "",
        f"- Baseline: `{report['baseline_variant']}`",
        f"- Candidate: `{report['candidate_variant']}`",
        f"- Paired tasks: {report['paired_tasks']}",
        (
            f"- Bootstrap: {report['bootstrap']['samples']} task-level draws, "
            f"seed {report['bootstrap']['seed']}"
        ),
        "",
        "| Metric | Baseline | Candidate | Delta | Paired CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in report["metrics"].items():
        lines.append(
            "| {name} | {baseline} | {candidate} | {delta} | [{low}, {high}] |".format(
                name=name,
                baseline=_format(value["baseline"]),
                candidate=_format(value["candidate"]),
                delta=_format(value["delta"]),
                low=_format(value["ci_low"]),
                high=_format(value["ci_high"]),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _by_task(rows: list[TurnRecord]) -> dict[str, list[TurnRecord]]:
    result: dict[str, list[TurnRecord]] = {}
    for row in rows:
        result.setdefault(row.task_id, []).append(row)
    return result


def _resample(
    by_task: Mapping[str, list[TurnRecord]], sampled: list[str]
) -> list[TurnRecord]:
    result: list[TurnRecord] = []
    for draw_index, task_id in enumerate(sampled):
        result.extend(
            replace(row, task_id=f"{task_id}#bootstrap-{draw_index}")
            for row in by_task[task_id]
        )
    return result


def _metric_value(value: Mapping[str, Any], dotted: str) -> float | None:
    if dotted in value:
        current: Any = value[dotted]
    elif "." in dotted:
        head, tail = dotted.split(".", 1)
        nested = value.get(head)
        if not isinstance(nested, Mapping) or tail not in nested:
            return None
        current = nested[tail]
    else:
        return None
    return float(current) if isinstance(current, (int, float)) else None


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _format(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"
