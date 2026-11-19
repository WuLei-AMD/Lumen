"""Validated launcher/config builder; it never starts subprocesses itself."""

from __future__ import annotations

import os
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from geak_agent_coder.config import load_composed_config, validate_training_config


def load_config(path: str | Path) -> dict[str, Any]:
    config = load_composed_config(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    validate_training_config(config)


def assert_training_authorized(config: dict[str, Any]) -> None:
    """Verify the pinned data-build artifact and any recipe-specific gates."""

    execution = config.get("execution", {})
    if execution.get("allow_training") is not True:
        raise RuntimeError(
            "training is disabled; set execution.allow_training=true only after "
            "the final Phase 1 dataset is frozen"
        )
    revision = config.get("model", {}).get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise RuntimeError("model.revision must pin an immutable HF revision")
    for key in ("tokenizer_sha256", "data_manifest_sha256"):
        value = execution.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise RuntimeError(f"execution.{key} must be a 64-character SHA256")
    manifest_value = execution.get("data_manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise RuntimeError("execution.data_manifest_path must be pinned")
    manifest_path = Path(manifest_value).expanduser().resolve()
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read pinned data manifest: {manifest_path}") from exc
    actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_hash != execution["data_manifest_sha256"].lower():
        raise RuntimeError(
            "data manifest SHA256 mismatch: "
            f"expected {execution['data_manifest_sha256']}, got {actual_manifest_hash}"
        )
    if manifest.get("schema_version") != "geak_training_data_manifest_v2":
        raise RuntimeError("training requires a geak_training_data_manifest_v2 build")
    tokenizer = manifest.get("tokenizer", {})
    if tokenizer.get("sha256") != execution["tokenizer_sha256"].lower():
        raise RuntimeError("training tokenizer SHA256 does not match data manifest")
    train_artifact = manifest.get("artifacts", {}).get("train", {})
    artifact_name, expected_train_hash = (
        train_artifact.get("path"),
        train_artifact.get("sha256"),
    )
    if not isinstance(artifact_name, str) or not isinstance(expected_train_hash, str):
        raise RuntimeError("data manifest has no pinned training artifact")
    artifact_path = (manifest_path.parent / artifact_name).resolve()
    configured_train = Path(config["data"]["train_path"]).expanduser().resolve()
    try:
        artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        configured_hash = hashlib.sha256(configured_train.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError("pinned tokenized training artifact is unavailable") from exc
    if artifact_hash != expected_train_hash or configured_hash != expected_train_hash:
        raise RuntimeError("configured training data does not match the pinned manifest")
    configured_dev_value = config["data"].get("validation_path")
    if configured_dev_value:
        dev_artifact = manifest.get("artifacts", {}).get("dev", {})
        dev_name, expected_dev_hash = (
            dev_artifact.get("path"),
            dev_artifact.get("sha256"),
        )
        if not isinstance(dev_name, str) or not isinstance(expected_dev_hash, str):
            raise RuntimeError("data manifest has no pinned validation artifact")
        manifest_dev = (manifest_path.parent / dev_name).resolve()
        configured_dev = Path(configured_dev_value).expanduser().resolve()
        try:
            manifest_dev_hash = hashlib.sha256(manifest_dev.read_bytes()).hexdigest()
            configured_dev_hash = hashlib.sha256(configured_dev.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError("pinned tokenized validation artifact is unavailable") from exc
        if (
            manifest_dev_hash != expected_dev_hash
            or configured_dev_hash != expected_dev_hash
        ):
            raise RuntimeError(
                "configured validation data does not match the pinned manifest"
            )

    recipe_name = config.get("recipe", {}).get("name")
    if recipe_name == "geak_mi308_release_v1":
        replay_root = execution.get("replay_package_root")
        if not isinstance(replay_root, str) or not replay_root.strip():
            raise RuntimeError("GEAK release requires execution.replay_package_root")
        from geak_agent_coder.data import enforce_final_replay_gates

        gate = enforce_final_replay_gates(replay_root)
        replay_sources = [
            source
            for source in manifest.get("sources", [])
            if isinstance(source, dict) and source.get("policy") == "geak_replay_release"
        ]
        if not gate.ready or not any(
            source.get("replay_gate", {}).get("ready") is True
            for source in replay_sources
        ):
            raise RuntimeError("GEAK replay release gate is absent or not ready")


def build_torchrun_command(
    config_path: str | Path,
    *,
    python_executable: str | None = None,
) -> list[str]:
    config = load_config(config_path)
    distributed = config["distributed"]
    hardware = config.get("hardware", {})
    nodes = hardware.get("nodes", 1)
    executable = python_executable or os.environ.get("PYTHON", sys.executable)
    command = [
        executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nodes}",
        f"--nproc-per-node={distributed['nproc_per_node']}",
    ]
    if nodes == 1:
        command.append("--standalone")
    else:
        rendezvous = distributed.get("rendezvous", {})
        endpoint = rendezvous.get("endpoint") if isinstance(rendezvous, dict) else None
        run_id = rendezvous.get("run_id") if isinstance(rendezvous, dict) else None
        if not endpoint or not run_id:
            raise ValueError(
                "multi-node launch requires distributed.rendezvous.endpoint and run_id"
            )
        command.extend(
            [
                "--rdzv-backend=c10d",
                f"--rdzv-endpoint={endpoint}",
                f"--rdzv-id={run_id}",
            ]
        )
    command.extend(
        [
            "-m",
            "geak_agent_coder.sft.train",
            "--config",
            str(Path(config_path)),
        ]
    )
    return command


def format_torchrun_command(config_path: str | Path) -> str:
    return shlex.join(build_torchrun_command(config_path))
