from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from geak_agent_coder.data import (
    DataContractError,
    TrainingSplitError,
    build_from_config,
    path_sha256,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def kernel(sample_id: str, split: str = "train") -> dict:
    return {
        "schema_version": "geak_kernel_sft_v1",
        "sample_id": sample_id,
        "sample_domain": "kernel",
        "task_type": "cold_start",
        "split": split,
        "input": {"contract": {"language": "python"}, "parent_source": "x = 1"},
        "output": {"patch": "PATCH"},
        "labels": {
            "patch_applies": True,
            "compile_pass": True,
            "correctness_pass": True,
            "benchmark_valid": True,
        },
        "provenance": {"source_hash": sample_id, "lane": "test"},
    }


def replay(sample_id: str) -> dict:
    return {
        "schema_version": "general_coding_replay_v1",
        "sample_id": sample_id,
        "sample_domain": "general_coding",
        "task_type": "general_coding_replay",
        "coding_task_type": "bug_fix",
        "primary_language": "python",
        "split": "train",
        "input": {"problem_statement": "Fix it"},
        "output": {"patch": "PATCH"},
        "provenance": {"dataset_id": "local", "dataset_revision": "abc"},
    }


def generic(sample_id: str, split: str = "train") -> dict:
    return {
        "id": sample_id,
        "split": split,
        "prompt": "Implement a safe parser",
        "response": "def parse(value):\n    return value",
        "dataset_id": "custom/coding",
        "dataset_revision": "abc123",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def canonical_mapping(*, replay_rows: bool = False) -> dict:
    fields = [
        "schema_version",
        "sample_id",
        "sample_domain",
        "task_type",
        "split",
        "input",
        "output",
        "provenance",
    ]
    if replay_rows:
        fields += ["coding_task_type", "primary_language"]
    else:
        fields += ["labels"]
    return {field: {"source": field} for field in fields}


def write_release(root: Path, rows: list[dict]) -> None:
    root.mkdir()
    write_jsonl(root / "accepted.jsonl", rows)
    (root / "leakage-report.json").write_text(
        json.dumps({"passed": True, "train_eval_overlap": 0}), encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps({"status": "ready"}), encoding="utf-8"
    )
    (root / "trust-report.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "offline_cpu_only_double_replay_passed": len(rows),
                "quality_failures": 0,
            }
        ),
        encoding="utf-8",
    )
    checksums = []
    for name in (
        "accepted.jsonl",
        "leakage-report.json",
        "manifest.json",
        "trust-report.json",
    ):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        checksums.append(f"{digest}  {name}\n")
    (root / "checksums.sha256").write_text("".join(checksums), encoding="utf-8")


def make_config(
    tmp_path: Path,
    *,
    kernel_rows: list[dict],
    replay_rows: list[dict] | None = None,
    mix_enabled: bool = False,
) -> Path:
    kernel_path = tmp_path / "kernel.jsonl"
    write_jsonl(kernel_path, kernel_rows)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    sources = [
        {
            "name": "kernel",
            "type": "local_jsonl",
            "path": str(kernel_path),
            "admission": {
                "type": "generic",
                "allowed_splits": ["train", "dev"],
                "required_provenance_fields": ["source_hash"],
            },
            "field_mapping": canonical_mapping(),
        }
    ]
    if replay_rows is not None:
        release = tmp_path / "replay"
        write_release(release, replay_rows)
        sources.append(
            {
                "name": "replay",
                "type": "local_jsonl",
                "path": str(release / "accepted.jsonl"),
                "admission": {
                    "type": "geak_replay_release",
                    "package_root": str(release),
                    "accepted_rows": len(replay_rows),
                    "allowed_splits": ["train"],
                    "required_provenance_fields": [
                        "dataset_id",
                        "dataset_revision",
                    ],
                },
                "field_mapping": canonical_mapping(replay_rows=True),
            }
        )
    config = {
        "schema_version": "geak_data_config_v2",
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": path_sha256(tokenizer_path),
        },
        "tokenization": {"max_length": 10000},
        "sampling": {
            "steps": 6,
            "seed": 7,
            "replay_mix": {
                "enabled": mix_enabled,
                "min_assistant_loss_token_share": 0.15,
                "max_assistant_loss_token_share": 0.20,
            },
        },
        "sources": sources,
        "output": {"directory": str(tmp_path / "output")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_multi_source_build_materializes_schedule_and_dev_isolation(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        kernel_rows=[kernel("k-train"), kernel("k-dev", "dev")],
        replay_rows=[replay("r-train")],
        mix_enabled=True,
    )
    manifest = build_from_config(config, tokenizer=FakeTokenizer())
    output = tmp_path / "output"
    train = [
        json.loads(line)
        for line in (output / "train.tokenized.jsonl").read_text().splitlines()
    ]
    dev = [
        json.loads(line)
        for line in (output / "dev.tokenized.jsonl").read_text().splitlines()
    ]
    assert len(train) == 6
    assert {row["sample_id"] for row in train} == {"k-train", "r-train"}
    assert [row["sample_id"] for row in dev] == ["k-dev"]
    assert manifest["sampling"]["indices"]
    assert manifest["artifacts"]["train"]["rows"] == 6
    assert manifest["dev"]["source_rows"] == 1
    assert (output / "manifest.json").stat().st_mode & 0o222 == 0


def test_single_domain_build_allowed_only_when_mix_disabled(tmp_path: Path) -> None:
    config = make_config(tmp_path, kernel_rows=[kernel("k")])
    manifest = build_from_config(config, tokenizer=FakeTokenizer())
    assert manifest["sampling"]["sample_counts"] == {"kernel": 6}


def test_split_isolation_and_replay_gate_bypass_prevention(tmp_path: Path) -> None:
    config = make_config(tmp_path, kernel_rows=[kernel("held", "test")])
    with pytest.raises(TrainingSplitError, match="not admitted"):
        build_from_config(config, tokenizer=FakeTokenizer())

    replay_path = tmp_path / "kernel.jsonl"
    write_jsonl(replay_path, [replay("smuggled")])
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    loaded["sources"][0]["field_mapping"] = canonical_mapping(replay_rows=True)
    loaded["sources"][0]["admission"]["allowed_splits"] = ["train"]
    config.write_text(yaml.safe_dump(loaded), encoding="utf-8")
    with pytest.raises(DataContractError, match="require geak_replay_release"):
        build_from_config(config, tokenizer=FakeTokenizer())


def test_strict_replay_policy_cannot_gate_a_different_source(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        kernel_rows=[kernel("k")],
        replay_rows=[replay("released")],
        mix_enabled=True,
    )
    rogue = tmp_path / "rogue.jsonl"
    write_jsonl(rogue, [replay("ungated")])
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    loaded["sources"][1]["path"] = str(rogue)
    config.write_text(yaml.safe_dump(loaded), encoding="utf-8")
    with pytest.raises(DataContractError, match="gated accepted artifact"):
        build_from_config(config, tokenizer=FakeTokenizer())


def test_generic_raw_coding_rows_do_not_require_geak_replay_gate(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "generic.jsonl"
    write_jsonl(source_path, [generic("g-train"), generic("g-dev", "dev")])
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = {
        "schema_version": "geak_data_config_v2",
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": path_sha256(tokenizer_path),
        },
        "tokenization": {"max_length": 10000},
        "sampling": {"steps": 3, "replay_mix": {"enabled": False}},
        "sources": [
            {
                "name": "generic",
                "type": "local_jsonl",
                "path": str(source_path),
                "admission": {
                    "type": "generic",
                    "allowed_splits": ["train", "dev"],
                    "required_provenance_fields": [
                        "dataset_id",
                        "dataset_revision",
                    ],
                },
                "field_mapping": {
                    "schema_version": {"const": "generic_coding_sft_v1"},
                    "sample_id": {"source": "id"},
                    "sample_domain": {"const": "generic_coding"},
                    "task_type": {"const": "coding_instruction"},
                    "split": {"source": "split"},
                    "input": {
                        "object": {"prompt": {"source": "prompt"}}
                    },
                    "output": {
                        "object": {"response": {"source": "response"}}
                    },
                    "provenance": {
                        "object": {
                            "dataset_id": {"source": "dataset_id"},
                            "dataset_revision": {
                                "source": "dataset_revision"
                            },
                        }
                    },
                },
            }
        ],
        "output": {"directory": str(tmp_path / "generic-output")},
    }
    config_path = tmp_path / "generic-config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = build_from_config(config_path, tokenizer=FakeTokenizer())
    assert manifest["sampling"]["sample_counts"] == {"generic_coding": 3}
    dev_rows = (tmp_path / "generic-output/dev.tokenized.jsonl").read_text()
    assert '"sample_id":"g-dev"' in dev_rows
