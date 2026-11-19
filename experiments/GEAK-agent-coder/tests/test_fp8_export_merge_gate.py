import json

import pytest
import torch

from geak_agent_coder.export.exporter import FORMAT, mark_deployable
from geak_agent_coder.export.merge import adapter_updates, merge_updates_into_state


def test_merge_attention_and_ep_local_packed_expert():
    adapters = {
        "model.layers.0.self_attn.q_proj.lora_A": torch.ones(2, 4),
        "model.layers.0.self_attn.q_proj.lora_B": torch.ones(3, 2),
        "model.layers.0.mlp.local_experts.experts.1.gate_up_proj.lora_A": torch.ones(1, 4),
        "model.layers.0.mlp.local_experts.experts.1.gate_up_proj.lora_B": torch.ones(6, 1),
    }
    metadata = {
        "expert_parallel_size": 2,
        "experts_per_rank": 2,
        "modules": {
            "model.layers.0.self_attn.q_proj": {"rank": 2, "alpha": 4},
            "model.layers.0.mlp.local_experts.experts.1.gate_up_proj": {
                "rank": 1,
                "alpha": 2,
            },
        },
    }
    updates = adapter_updates(adapters, metadata, rank=1)
    state = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(3, 4),
        "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(4, 6, 4),
    }
    merged = merge_updates_into_state(state, updates)

    assert torch.all(merged["model.layers.0.self_attn.q_proj.weight"] == 4)
    packed = merged["model.layers.0.mlp.experts.gate_up_proj"]
    assert torch.count_nonzero(packed[:3]) == 0
    assert torch.all(packed[3] == 2)


def test_deployable_gate_rejects_missing_smoke_evidence(tmp_path):
    (tmp_path / "fp8_manifest.json").write_text(
        json.dumps(
            {
                "deployment_status": "candidate",
                "vllm_smoke_validated": False,
                "artifact_sha256": "a" * 64,
                "target_arch": "gfx942",
                "fp8_format": "e4m3fnuz",
            }
        )
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"engine": "vllm", "status": "not_run"}))
    with pytest.raises(ValueError, match="requires passed vLLM smoke"):
        mark_deployable(tmp_path, evidence)
    assert not (tmp_path / "DEPLOYABLE").exists()


def test_deployable_gate_accepts_explicit_passed_evidence(tmp_path):
    (tmp_path / "fp8_manifest.json").write_text(
        json.dumps(
            {
                "deployment_status": "candidate",
                "vllm_smoke_validated": False,
                "artifact_sha256": "a" * 64,
                "target_arch": "gfx942",
                "fp8_format": "e4m3fnuz",
            }
        )
    )
    evidence = tmp_path / "smoke.json"
    evidence.write_text(
        json.dumps(
            {
                "engine": "vllm",
                "status": "passed",
                "artifact_format": FORMAT,
                "artifact_sha256": "a" * 64,
                "device_arch": "gfx942",
                "fp8_format": "e4m3fnuz",
                "command": "vllm serve /artifact",
                "timestamp": "2026-09-21T00:00:00Z",
            }
        )
    )
    manifest = mark_deployable(tmp_path, evidence)
    assert manifest["deployment_status"] == "deployable"
    assert manifest["vllm_smoke_validated"] is True
    assert (tmp_path / "DEPLOYABLE").exists()
