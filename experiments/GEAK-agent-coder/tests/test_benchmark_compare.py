import json
from dataclasses import replace

import pytest

from geak_agent_coder.benchmark.compare import (
    assert_comparable_campaigns,
    compare_variants,
    write_comparison_report,
)
from geak_agent_coder.benchmark.models import FixedBenchmarkIdentity, TurnRecord


IDENTITY = FixedBenchmarkIdentity("agent", "suite", "decode")


def make_records(variant: str, passed: list[bool]) -> list[TurnRecord]:
    return [
        TurnRecord(
            task_id=f"task-{index}",
            variant=variant,
            seed=7,
            turn=1,
            compiled=value,
            correct=value,
            speedup_vs_frozen_baseline=1.5 if value else 0.0,
            tokens_input=100,
            tokens_output=20,
            wall_time_s=2.0,
            error_type=None if value else "compile",
            operator="gemm",
            lane="triton",
        )
        for index, value in enumerate(passed)
    ]


def test_paired_bootstrap_is_deterministic_and_positive() -> None:
    baseline = make_records("base", [True, False, False, True])
    candidate = make_records("sft", [True, True, True, True])

    first = compare_variants(
        baseline,
        candidate,
        fixed_identity=IDENTITY,
        bootstrap_samples=100,
        bootstrap_seed=42,
    )
    second = compare_variants(
        baseline,
        candidate,
        fixed_identity=IDENTITY,
        bootstrap_samples=100,
        bootstrap_seed=42,
    )

    assert first == second
    assert first["metrics"]["pass_rate"]["delta"] == 0.5
    assert first["metrics"]["fast_at_p.1.2"]["delta"] == 0.5
    assert first["metrics"]["pass_rate"]["ci_low"] >= 0.0
    assert first["bootstrap"]["unit"] == "task"


def test_report_writes_json_and_dependency_free_markdown(tmp_path) -> None:
    report = compare_variants(
        make_records("base", [False, True]),
        make_records("sft", [True, True]),
        fixed_identity=IDENTITY,
        bootstrap_samples=10,
    )
    json_path, markdown_path = write_comparison_report(report, tmp_path)

    assert json.loads(json_path.read_text())["paired_tasks"] == 2
    markdown = markdown_path.read_text()
    assert "# GEAK agent-loop comparison" in markdown
    assert "pass_rate" in markdown


def test_fixed_hashes_and_pairing_fail_closed() -> None:
    baseline_manifest = {"fixed_identity": IDENTITY.to_dict()}
    candidate_manifest = {"fixed_identity": IDENTITY.to_dict()}
    assert assert_comparable_campaigns(baseline_manifest, candidate_manifest) == IDENTITY

    candidate_manifest["fixed_identity"] = {
        **IDENTITY.to_dict(),
        "decode_hash": "different",
    }
    with pytest.raises(ValueError, match="decode_hash"):
        assert_comparable_campaigns(baseline_manifest, candidate_manifest)

    candidate = make_records("sft", [True])
    candidate[0] = replace(candidate[0], task_id="other")
    with pytest.raises(ValueError, match="identical task IDs"):
        compare_variants(
            make_records("base", [True]),
            candidate,
            fixed_identity=IDENTITY,
            bootstrap_samples=2,
        )

    with pytest.raises(ValueError, match="task/seed pairs"):
        compare_variants(
            make_records("base", [True]),
            [replace(make_records("sft", [True])[0], seed=99)],
            fixed_identity=IDENTITY,
            bootstrap_samples=2,
        )
