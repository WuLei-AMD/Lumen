from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from geak_agent_coder.data import (
    ReplayGateError,
    SchemaValidationError,
    TrainingSplitError,
    enforce_final_replay_gates,
    load_samples,
    load_training_samples,
    validate_sample,
)


def kernel_sample(sample_id: str = "kernel-1", split: str = "train") -> dict:
    return {
        "schema_version": "geak_kernel_sft_v1",
        "sample_id": sample_id,
        "sample_domain": "kernel",
        "task_type": "direction_conditioned",
        "split": split,
        "input": {"contract": {}, "parent_source": "x = 1"},
        "output": {"patch": "--- a/x.py\n+++ b/x.py\n@@\n-x = 1\n+x = 2\n"},
        "labels": {
            "patch_applies": True,
            "compile_pass": True,
            "correctness_pass": True,
            "benchmark_valid": True,
        },
        "provenance": {"source_hash": "abc"},
    }


def replay_sample(sample_id: str = "replay-1", split: str = "train") -> dict:
    return {
        "schema_version": "general_coding_replay_v1",
        "sample_id": sample_id,
        "sample_domain": "general_coding",
        "task_type": "general_coding_replay",
        "coding_task_type": "function_implementation",
        "split": split,
        "input": {"problem_statement": "Implement f.", "parent_source": "pass"},
        "output": {"patch": "--- a/f.py\n+++ b/f.py\n@@\n-pass\n+return 1\n"},
        "provenance": {"dataset_id": "local", "dataset_revision": "deadbeef"},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_loads_hf_layout_directory_and_dataset_like(tmp_path: Path) -> None:
    root = tmp_path / "hf"
    root.mkdir()
    rows = [kernel_sample(), replay_sample()]
    write_jsonl(root / "samples.jsonl", rows)

    assert [row["sample_id"] for row in load_samples(root)] == [
        "kernel-1",
        "replay-1",
    ]

    class FakeDataset:
        def __iter__(self):
            return iter(rows)

    assert load_samples(FakeDataset()) == rows


def test_schema_validation_is_fail_closed() -> None:
    bad_kernel = kernel_sample()
    del bad_kernel["labels"]["correctness_pass"]
    with pytest.raises(SchemaValidationError, match="correctness_pass"):
        validate_sample(bad_kernel)

    bad_replay = replay_sample()
    bad_replay["task_type"] = "direction_conditioned"
    with pytest.raises(SchemaValidationError, match="general_coding_replay"):
        validate_sample(bad_replay)

    unknown = kernel_sample()
    unknown["schema_version"] = "future_v2"
    with pytest.raises(SchemaValidationError, match="schema_version"):
        validate_sample(unknown)


@pytest.mark.parametrize("split", ["dev", "held_out", "held-out", "test"])
def test_training_loader_rejects_eval_splits(tmp_path: Path, split: str) -> None:
    path = tmp_path / "samples.jsonl"
    write_jsonl(path, [kernel_sample(), replay_sample(split=split)])
    with pytest.raises(TrainingSplitError, match=split):
        load_training_samples(path)


def _write_release(root: Path, *, count: int = 500, leakage: bool = True) -> None:
    root.mkdir()
    accepted = root / "accepted.jsonl"
    write_jsonl(accepted, [{"sample_id": f"r-{index}"} for index in range(count)])
    (root / "leakage-report.json").write_text(
        json.dumps({"passed": leakage, "train_eval_overlap": 0 if leakage else 1}),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"status": "ready"}), encoding="utf-8"
    )
    (root / "trust-report.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "offline_cpu_only_double_replay_passed": count,
                "quality_failures": 0,
            }
        ),
        encoding="utf-8",
    )
    targets = [
        "accepted.jsonl",
        "leakage-report.json",
        "manifest.json",
        "trust-report.json",
    ]
    lines = []
    for name in targets:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    (root / "checksums.sha256").write_text("".join(lines), encoding="utf-8")


def test_final_replay_gates_verify_count_reports_and_checksums(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release"
    _write_release(root)
    result = enforce_final_replay_gates(root)
    assert result.ready
    assert result.accepted_count == 500
    assert result.trust_passed
    assert result.checksums_verified == 4


def test_final_replay_gates_reject_wrong_count_and_leakage(tmp_path: Path) -> None:
    root = tmp_path / "release"
    _write_release(root, count=499, leakage=False)
    with pytest.raises(ReplayGateError, match="accepted=499"):
        enforce_final_replay_gates(root)


def test_final_replay_gates_reject_checksum_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "release"
    _write_release(root)
    with (root / "accepted.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ReplayGateError, match="checksum mismatch"):
        enforce_final_replay_gates(root)
