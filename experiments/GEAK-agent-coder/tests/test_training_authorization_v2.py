import hashlib
import json
from pathlib import Path

import pytest

from geak_agent_coder.sft.launcher import assert_training_authorized, load_config


ROOT = Path(__file__).parents[1]


def _authorized_generic_config(tmp_path):
    config = load_config(ROOT / "configs/sft/mi300_jsonl_fp8.yaml")
    train = tmp_path / "train.tokenized.jsonl"
    train.write_text('{"input_ids":[1,2],"loss_mask":[0,1]}\n', encoding="utf-8")
    train_sha = hashlib.sha256(train.read_bytes()).hexdigest()
    tokenizer_sha = "a" * 64
    manifest = {
        "schema_version": "geak_training_data_manifest_v2",
        "tokenizer": {"sha256": tokenizer_sha},
        "sources": [{"name": "custom", "policy": "generic"}],
        "artifacts": {
            "train": {
                "path": train.name,
                "sha256": train_sha,
                "rows": 1,
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    config["model"]["revision"] = "pinned-revision"
    config["data"]["train_path"] = str(train)
    config["data"]["validation_path"] = None
    config["execution"].update(
        allow_training=True,
        data_manifest_path=str(manifest_path),
        data_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        tokenizer_sha256=tokenizer_sha,
    )
    return config, train


def test_generic_recipe_authorizes_pinned_non_geak_dataset(tmp_path):
    config, _ = _authorized_generic_config(tmp_path)
    assert_training_authorized(config)


def test_authorization_rejects_training_artifact_tampering(tmp_path):
    config, train = _authorized_generic_config(tmp_path)
    train.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        assert_training_authorized(config)
