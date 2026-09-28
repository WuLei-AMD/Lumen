import json

import pytest

from geak_agent_coder.export.exporter import FORMAT, mark_deployable


def _write_candidate(tmp_path):
    (tmp_path / "fp8_manifest.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "target_arch": "gfx950",
                "fp8_format": "e4m3fn",
                "deployment_status": "candidate",
                "vllm_smoke_validated": False,
                "artifact_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )


def _write_evidence(tmp_path, **overrides):
    evidence = {
        "engine": "vllm",
        "status": "passed",
        "artifact_format": FORMAT,
        "artifact_sha256": "b" * 64,
        "device_arch": "gfx950",
        "fp8_format": "e4m3fn",
        "command": "vllm serve /artifact",
        "timestamp": "2026-09-21T00:00:00Z",
    }
    evidence.update(overrides)
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [("device_arch", "gfx942"), ("fp8_format", "e4m3fnuz")],
)
def test_deploy_gate_compares_hardware_evidence_to_manifest(
    tmp_path, field, wrong_value
):
    _write_candidate(tmp_path)
    evidence = _write_evidence(tmp_path, **{field: wrong_value})
    with pytest.raises(ValueError, match="mismatches"):
        mark_deployable(tmp_path, evidence)
    assert not (tmp_path / "DEPLOYABLE").exists()


def test_deploy_gate_accepts_manifest_hardware_and_format(tmp_path):
    _write_candidate(tmp_path)
    evidence = _write_evidence(tmp_path)
    manifest = mark_deployable(tmp_path, evidence)
    assert manifest["deployment_status"] == "deployable"
    assert manifest["target_arch"] == "gfx950"
    assert manifest["fp8_format"] == "e4m3fn"
