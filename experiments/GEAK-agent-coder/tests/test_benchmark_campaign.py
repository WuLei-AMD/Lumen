import json
from pathlib import Path

import pytest

from geak_agent_coder.benchmark.campaign import (
    canonical_hash,
    load_campaign,
    write_campaign_manifest,
)


def campaign_yaml(extra_variant: str = "", role: str = "engineer") -> str:
    return f"""
name: cpu-test
fixed_identity:
  agent_hash: agent
  suite_hash: suite
  decode_hash: decode
tasks_manifest: tasks.json
output_root: output
protected_sha256: ["{'a' * 64}"]
variants:
  - name: base
    base_url: http://base/v1
    model: base-model
    checkpoint_identity: base-checkpoint
    {extra_variant}
  - name: sft
    base_url: http://sft/v1
    model: sft-model
    checkpoint_identity: sft-checkpoint
runner:
  role: {role}
gpu_isolation:
  enabled: true
  execute: false
"""


def test_loads_campaign_and_writes_artifact_manifest(tmp_path) -> None:
    config = tmp_path / "campaign.yaml"
    config.write_text(campaign_yaml())
    campaign = load_campaign(config)

    assert campaign.tasks_manifest == tmp_path / "tasks.json"
    assert [variant.name for variant in campaign.variants] == ["base", "sft"]
    output = write_campaign_manifest(
        campaign,
        status="planned",
        artifacts={"turns": "turns.jsonl"},
        path=tmp_path / "manifest.json",
    )
    manifest = json.loads(output.read_text())
    assert manifest["fixed_identity"]["agent_hash"] == "agent"
    assert len(manifest["config_hash"]) == 64
    assert manifest["gpu_isolation"]["execute"] is False


def test_variants_cannot_smuggle_agent_or_decode_changes(tmp_path) -> None:
    config = tmp_path / "campaign.yaml"
    config.write_text(campaign_yaml("temperature: 0.9"))
    with pytest.raises(ValueError, match="unexpected keys"):
        load_campaign(config)


def test_primary_runner_is_engineer_only(tmp_path) -> None:
    config = tmp_path / "campaign.yaml"
    config.write_text(campaign_yaml(role="director"))
    with pytest.raises(ValueError, match="engineer-only"):
        load_campaign(config)


def test_canonical_hash_ignores_mapping_order() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_all_benchmark_schemas_are_valid_json() -> None:
    root = Path(__file__).parents[1] / "manifests" / "benchmark"
    schemas = list(root.glob("*.schema.json"))
    assert schemas
    for schema in schemas:
        value = json.loads(schema.read_text())
        assert value["$schema"].endswith("2020-12/schema")
