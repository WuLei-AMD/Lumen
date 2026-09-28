from pathlib import Path

import pytest

from geak_agent_coder.benchmark.models import TurnRecord
from geak_agent_coder.benchmark.trajectory import (
    TrajectoryFormatError,
    parse_trajectory,
)


FIXTURE = Path(__file__).parent / "fixtures" / "benchmark" / "trajectory.jsonl"


def test_parses_agent_loop_events_and_resets_per_turn_costs() -> None:
    records = parse_trajectory(FIXTURE, operator="gemm", lane="triton")

    assert len(records) == 2
    assert records[0] == TurnRecord(
        task_id="gemm-001",
        variant="base",
        seed=11,
        turn=1,
        compiled=False,
        correct=False,
        speedup_vs_frozen_baseline=0.0,
        tokens_input=100,
        tokens_output=20,
        wall_time_s=2.0,
        error_type="compile",
        operator="gemm",
        lane="triton",
        tool_calls=1,
    )
    assert records[1].passed
    assert records[1].speedup_vs_frozen_baseline == 1.5
    assert records[1].tokens_total == 150
    assert records[1].wall_time_s == 3.0


def test_parses_direct_turn_rows_with_missing_usage() -> None:
    records = parse_trajectory(
        [
            {
                "task_id": "softmax-1",
                "variant": "sft",
                "seed": 3,
                "turn": 1,
                "compiled": True,
                "correct": False,
                "speedup_vs_frozen_baseline": None,
            }
        ]
    )
    assert records[0].tokens_total == 0
    assert records[0].wall_time_s == 0.0


def test_rejects_missing_identity() -> None:
    with pytest.raises(TrajectoryFormatError, match="missing task_id"):
        parse_trajectory(
            [
                {
                    "turn": 1,
                    "compiled": True,
                    "correct": True,
                    "variant": "base",
                }
            ]
        )


def test_does_not_mislabel_torch_compile_metric() -> None:
    record = parse_trajectory(
        [
            {
                "task_id": "x",
                "variant": "base",
                "seed": 0,
                "turn": 1,
                "compiled": True,
                "correct": True,
                "speedup_vs_compile": 99.0,
            }
        ]
    )[0]
    assert record.speedup_vs_frozen_baseline is None


def test_preserves_assistant_turns_without_evaluation() -> None:
    records = parse_trajectory(
        [
            {
                "event": "agent_loop",
                "payload": {
                    "task_id": "x",
                    "variant": "base",
                    "seed": 0,
                    "events": [
                        {
                            "type": "model",
                            "elapsed_seconds": 1.0,
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                            },
                        },
                        {
                            "type": "tool",
                            "elapsed_seconds": 0.5,
                            "result": {"ok": True, "content": "source"},
                        },
                        {
                            "type": "model",
                            "elapsed_seconds": 2.0,
                            "usage": {
                                "prompt_tokens": 20,
                                "completion_tokens": 4,
                            },
                        },
                        {
                            "type": "tool",
                            "elapsed_seconds": 3.0,
                            "result": {
                                "evaluation": {
                                    "compiled": True,
                                    "correct": True,
                                    "speedup_geomean": 1.2,
                                }
                            },
                        },
                    ],
                },
            }
        ]
    )
    assert [record.turn for record in records] == [1, 2]
    assert not records[0].evaluated
    assert records[0].tokens_total == 12
    assert records[0].wall_time_s == 1.5
    assert records[1].evaluated and records[1].passed
    assert records[1].tokens_total == 24
