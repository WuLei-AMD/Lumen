from dataclasses import replace

import pytest

from geak_agent_coder.benchmark.metrics import aggregate_metrics
from geak_agent_coder.benchmark.models import TurnRecord


def row(
    task: str,
    seed: int,
    turn: int,
    *,
    compiled: bool,
    correct: bool,
    speedup: float | None,
    tokens: int = 10,
    operator: str = "gemm",
    lane: str = "triton",
) -> TurnRecord:
    return TurnRecord(
        task_id=task,
        variant="model",
        seed=seed,
        turn=turn,
        compiled=compiled,
        correct=correct,
        speedup_vs_frozen_baseline=speedup,
        tokens_input=tokens,
        wall_time_s=1.0,
        error_type=None if correct else ("compile" if not compiled else "correctness"),
        operator=operator,
        lane=lane,
    )


@pytest.fixture
def records() -> list[TurnRecord]:
    return [
        row("a", 0, 1, compiled=False, correct=False, speedup=0),
        row("a", 0, 2, compiled=True, correct=True, speedup=2.0, tokens=20),
        row("a", 1, 1, compiled=True, correct=True, speedup=1.3),
        row("a", 1, 2, compiled=True, correct=True, speedup=1.5),
        row(
            "b",
            0,
            1,
            compiled=True,
            correct=False,
            speedup=0,
            operator="softmax",
        ),
        row(
            "b",
            0,
            2,
            compiled=True,
            correct=False,
            speedup=0,
            operator="softmax",
        ),
        row(
            "b",
            1,
            1,
            compiled=False,
            correct=False,
            speedup=0,
            operator="softmax",
        ),
        row(
            "b",
            1,
            2,
            compiled=False,
            correct=False,
            speedup=0,
            operator="softmax",
        ),
    ]


def test_core_quality_and_cost_metrics(records: list[TurnRecord]) -> None:
    metrics = aggregate_metrics(records)

    assert metrics["pass_rate"] == 0.5
    assert metrics["pass_by_turn"] == {"1": 0.25, "2": 0.5}
    assert metrics["pass_at_k_seeds"] == {"1": 0.5, "2": 0.5}
    assert metrics["first_pass_rate"] == 0.25
    assert metrics["fast_at_p"]["1.2"] == 0.5
    assert metrics["turns_to_pass"] == 1.5
    assert metrics["tokens_cost_of_pass"] == 20.0
    assert metrics["time_cost_of_pass_s"] == 1.5
    assert metrics["speedup_auc"] == pytest.approx(0.6)
    assert metrics["geomean_speedup_vs_frozen_baseline"] == pytest.approx(3**0.5)


def test_error_trajectory_and_breakdown_metrics(records: list[TurnRecord]) -> None:
    metrics = aggregate_metrics(records)

    assert metrics["compile_error_rate"] == 3 / 8
    assert metrics["correctness_error_rate"] == 2 / 8
    assert metrics["monotonic_improvement_rate"] == 0.5
    assert metrics["regression_rate"] == 0.0
    assert set(metrics["by_operator"]) == {"gemm", "softmax"}
    assert metrics["by_operator"]["gemm"]["pass_rate"] == 1.0
    assert metrics["by_operator"]["softmax"]["pass_rate"] == 0.0
    assert set(metrics["by_lane"]) == {"triton"}


def test_rejects_mixed_variants_and_duplicate_turns(records: list[TurnRecord]) -> None:
    with pytest.raises(ValueError, match="one variant"):
        aggregate_metrics(records + [replace(records[0], task_id="z", variant="other")])

    duplicate = records + [records[0]]
    with pytest.raises(ValueError, match="duplicate turns"):
        aggregate_metrics(duplicate)


def test_empty_metrics_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate_metrics([])


def test_unevaluated_turn_counts_cost_but_not_compile_error() -> None:
    records = [
        TurnRecord(
            task_id="task",
            variant="model",
            seed=0,
            turn=1,
            compiled=False,
            correct=False,
            evaluated=False,
            tokens_input=10,
            tokens_output=2,
        ),
        TurnRecord(
            task_id="task",
            variant="model",
            seed=0,
            turn=2,
            compiled=True,
            correct=True,
            evaluated=True,
            speedup_vs_frozen_baseline=1.1,
            tokens_input=20,
            tokens_output=4,
        ),
    ]
    metrics = aggregate_metrics(records)
    assert metrics["compile_error_rate"] == 0.0
    assert metrics["evaluated_turn_rate"] == 0.5
    assert metrics["tokens_cost_of_pass"] == 36.0
    assert metrics["turns_to_pass"] == 2.0
