"""CPU-only GEAK agent-loop benchmark metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable, Sequence

from .models import TurnRecord


DEFAULT_FAST_THRESHOLDS = (1.0, 1.2, 1.5, 2.0)


def aggregate_metrics(
    records: Iterable[TurnRecord],
    *,
    fast_thresholds: Sequence[float] = DEFAULT_FAST_THRESHOLDS,
    include_breakdowns: bool = True,
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("cannot aggregate an empty trajectory")
    variants = {row.variant for row in rows}
    if len(variants) != 1:
        raise ValueError(f"aggregate one variant at a time, got {sorted(variants)}")

    runs = _group_runs(rows)
    max_turn = max(row.turn for row in rows)
    summaries = [_run_summary(run, max_turn) for run in runs.values()]
    successful = [item for item in summaries if item["passed"]]
    total_turns = len(rows)
    evaluated_rows = [row for row in rows if row.evaluated]
    correct_turns = sum(row.passed for row in evaluated_rows)
    compiled_turns = sum(row.compiled for row in evaluated_rows)
    best_speedups = [item["best_speedup"] for item in summaries]

    result: dict[str, Any] = {
        "variant": next(iter(variants)),
        "tasks": len({row.task_id for row in rows}),
        "seeds": sorted({row.seed for row in rows}),
        "runs": len(runs),
        "turn_records": total_turns,
        "pass_rate": _mean(item["passed"] for item in summaries),
        "pass_by_turn": {
            str(turn): _mean(item["passed_by_turn"][turn] for item in summaries)
            for turn in range(1, max_turn + 1)
        },
        "pass_at_k_seeds": _pass_at_k_by_seed(runs),
        "first_pass_rate": _mean(
            item["first_pass_turn"] == 1 for item in summaries
        ),
        "first_pass_faster_rate": _mean(
            item["first_pass_turn"] == 1 and item["best_at_first_pass"] > 1.0
            for item in summaries
        ),
        "fast_at_p": {
            _threshold_key(threshold): _mean(
                item["passed"] and item["best_speedup"] > threshold
                for item in summaries
            )
            for threshold in fast_thresholds
        },
        "geomean_speedup_vs_frozen_baseline": _geomean(
            [value for value in best_speedups if value > 0.0]
        ),
        "turns_to_pass": _mean(
            item["first_pass_turn"] for item in successful
        ),
        "turns_to_best": _mean(item["turn_to_best"] for item in successful),
        "tokens_cost_of_pass": _mean(
            item["tokens_to_pass"] for item in successful
        ),
        "time_cost_of_pass_s": _mean(
            item["time_to_pass_s"] for item in successful
        ),
        "tool_calls_cost_of_pass": _mean(
            item["tool_calls_to_pass"] for item in successful
        ),
        "evaluated_turn_rate": len(evaluated_rows) / total_turns,
        "compile_error_rate": _mean(not row.compiled for row in evaluated_rows),
        "correctness_error_rate": _mean(
            row.compiled and not row.correct for row in evaluated_rows
        ),
        "other_error_rate": _mean(
            bool(row.error_type)
            and row.error_type not in {"compile", "correctness"}
            for row in evaluated_rows
        ),
        "compiled_turn_rate": (
            compiled_turns / len(evaluated_rows) if evaluated_rows else None
        ),
        "correct_turn_rate": (
            correct_turns / len(evaluated_rows) if evaluated_rows else None
        ),
        "monotonic_improvement_rate": _pooled_rate(
            summaries, "monotonic_improvements", "adjacent_pairs"
        ),
        "regression_rate": _pooled_rate(
            summaries, "regressions", "adjacent_pairs"
        ),
        "speedup_auc": _mean(item["speedup_auc"] for item in summaries),
    }
    if include_breakdowns:
        result["by_operator"] = _breakdown(rows, "operator", fast_thresholds)
        result["by_lane"] = _breakdown(rows, "lane", fast_thresholds)
    return result


def _group_runs(
    rows: Iterable[TurnRecord],
) -> dict[tuple[str, int], list[TurnRecord]]:
    grouped: dict[tuple[str, int], list[TurnRecord]] = defaultdict(list)
    for row in rows:
        grouped[(row.task_id, row.seed)].append(row)
    for run in grouped.values():
        run.sort(key=lambda row: row.turn)
        turns = [row.turn for row in run]
        if len(turns) != len(set(turns)):
            raise ValueError(f"duplicate turns in run: {turns}")
    return dict(grouped)


def _run_summary(run: list[TurnRecord], max_turn: int) -> dict[str, Any]:
    first_pass_index = next(
        (index for index, row in enumerate(run) if row.passed), None
    )
    correct_speedups = [
        (row.turn, float(row.speedup_vs_frozen_baseline or 0.0))
        for row in run
        if row.passed
    ]
    best_turn, best_speedup = max(
        correct_speedups, key=lambda item: item[1], default=(0, 0.0)
    )
    scores = [
        float(row.speedup_vs_frozen_baseline or 0.0) if row.passed else 0.0
        for row in run
        if row.evaluated
    ]
    best_so_far = 0.0
    curve: list[float] = []
    by_turn = {row.turn: row.passed for row in run}
    for turn in range(1, max_turn + 1):
        row = next((item for item in run if item.turn == turn), None)
        if row and row.passed:
            best_so_far = max(
                best_so_far, float(row.speedup_vs_frozen_baseline or 0.0)
            )
        curve.append(best_so_far)
    adjacent = list(zip(scores, scores[1:]))
    if first_pass_index is None:
        prefix: list[TurnRecord] = []
        first_pass_turn = None
        best_at_first = 0.0
    else:
        prefix = run[: first_pass_index + 1]
        first = run[first_pass_index]
        first_pass_turn = first.turn
        best_at_first = float(first.speedup_vs_frozen_baseline or 0.0)
    return {
        "passed": first_pass_index is not None,
        "first_pass_turn": first_pass_turn,
        "best_at_first_pass": best_at_first,
        "best_speedup": best_speedup,
        "turn_to_best": best_turn or None,
        "tokens_to_pass": sum(row.tokens_total for row in prefix),
        "time_to_pass_s": sum(row.wall_time_s for row in prefix),
        "tool_calls_to_pass": sum(row.tool_calls for row in prefix),
        "monotonic_improvements": sum(right > left for left, right in adjacent),
        "regressions": sum(right < left for left, right in adjacent),
        "adjacent_pairs": len(adjacent),
        # Mean best-so-far speedup over the fixed turn budget is normalized AUC.
        "speedup_auc": _mean(curve),
        "passed_by_turn": {
            turn: any(
                passed for item_turn, passed in by_turn.items() if item_turn <= turn
            )
            for turn in range(1, max_turn + 1)
        },
    }


def _pass_at_k_by_seed(
    runs: dict[tuple[str, int], list[TurnRecord]]
) -> dict[str, float]:
    task_outcomes: dict[str, list[bool]] = defaultdict(list)
    for (task_id, _seed), run in runs.items():
        task_outcomes[task_id].append(any(row.passed for row in run))
    max_samples = min(len(values) for values in task_outcomes.values())
    return {
        str(k): _mean(
            _unbiased_pass_at_k(len(values), sum(values), k)
            for values in task_outcomes.values()
            if len(values) >= k
        )
        for k in range(1, max_samples + 1)
    }


def _unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if not 1 <= k <= n:
        raise ValueError("Pass@k requires 1 <= k <= number of samples")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _breakdown(
    rows: list[TurnRecord], field: str, thresholds: Sequence[float]
) -> dict[str, Any]:
    grouped: dict[str, list[TurnRecord]] = defaultdict(list)
    for row in rows:
        grouped[str(getattr(row, field))].append(row)
    return {
        name: aggregate_metrics(
            values, fast_thresholds=thresholds, include_breakdowns=False
        )
        for name, values in sorted(grouped.items())
    }


def _pooled_rate(
    summaries: list[dict[str, Any]], numerator: str, denominator: str
) -> float | None:
    total = sum(item[denominator] for item in summaries)
    if total == 0:
        return None
    return sum(item[numerator] for item in summaries) / total


def _mean(values: Iterable[Any]) -> float | None:
    items = [float(value) for value in values if value is not None]
    return fmean(items) if items else None


def _geomean(values: list[float]) -> float | None:
    return math.exp(fmean(math.log(value) for value in values)) if values else None


def _threshold_key(value: float) -> str:
    return f"{value:g}"
