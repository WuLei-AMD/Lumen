"""Configurable, fail-closed source and row admission policies."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    DataContractError,
    TrainingSplitError,
    enforce_final_replay_gates,
)


def path_sha256(path: str | Path) -> str:
    """Hash one file or a directory tree with stable relative-path framing."""

    target = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(target.read_bytes())
        return digest.hexdigest()
    if not target.is_dir():
        raise DataContractError(f"checksum target does not exist: {target}")
    files = sorted(item for item in target.rglob("*") if item.is_file())
    if not files:
        raise DataContractError(f"checksum target contains no files: {target}")
    for item in files:
        relative = item.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _allowed_splits(policy: Mapping[str, Any]) -> set[str]:
    configured = policy.get("allowed_splits", ["train", "dev"])
    if (
        not isinstance(configured, Sequence)
        or isinstance(configured, (str, bytes))
        or not configured
        or not all(isinstance(value, str) and value for value in configured)
    ):
        raise DataContractError("admission.allowed_splits must be non-empty strings")
    return {value.lower() for value in configured}


def verify_source_admission(
    source: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    policy_type = policy.get("type", "generic")
    if policy_type not in {"generic", "geak_replay_release"}:
        raise DataContractError(f"unknown admission policy: {policy_type!r}")
    identity: dict[str, Any] = {"policy": policy_type}
    source_path = source.get("path")
    if isinstance(source_path, str):
        identity["source_sha256"] = path_sha256(source_path)
    expected = policy.get("expected_sha256")
    if expected is not None:
        if not isinstance(expected, str) or len(expected) != 64:
            raise DataContractError("admission.expected_sha256 must be a SHA256")
        checksum_target = policy.get("checksum_path", source.get("path"))
        if not isinstance(checksum_target, str):
            raise DataContractError("checksum admission requires a local checksum_path")
        actual = path_sha256(checksum_target)
        if actual != expected.lower():
            raise DataContractError(
                f"source checksum mismatch: expected {expected.lower()}, got {actual}"
            )
        identity["admission_checksum_sha256"] = actual
    elif policy.get("require_checksum", False):
        raise DataContractError("admission requires expected_sha256")

    if policy_type == "geak_replay_release":
        package_root = policy.get("package_root")
        if not isinstance(package_root, str) or not package_root:
            raise DataContractError(
                "geak_replay_release admission requires package_root"
            )
        required = policy.get("accepted_rows", 500)
        if not isinstance(required, int) or isinstance(required, bool) or required <= 0:
            raise DataContractError("accepted_rows must be a positive integer")
        accepted_file = policy.get("accepted_file", "accepted.jsonl")
        if not isinstance(accepted_file, str) or not accepted_file:
            raise DataContractError("accepted_file must be a non-empty string")
        if source.get("type") != "local_jsonl" or not isinstance(source_path, str):
            raise DataContractError(
                "geak_replay_release must use its accepted local_jsonl artifact"
            )
        accepted_path = (
            Path(package_root).expanduser().resolve() / accepted_file
        ).resolve()
        if Path(source_path).expanduser().resolve() != accepted_path:
            raise DataContractError(
                "replay source path must be the gated accepted artifact: "
                f"{accepted_path}"
            )
        gate = enforce_final_replay_gates(
            package_root,
            required_accepted_count=required,
            accepted_file=accepted_file,
        )
        identity["replay_gate"] = {
            "accepted_count": gate.accepted_count,
            "required_accepted_count": gate.required_accepted_count,
            "checksums_verified": gate.checksums_verified,
            "ready": gate.ready,
        }
        identity["package_sha256"] = path_sha256(package_root)
    return identity


def admit_row(row: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    split = str(row.get("split", "")).lower()
    if split not in _allowed_splits(policy):
        raise TrainingSplitError(
            f"{row.get('sample_id', '<unknown>')}: split {split!r} is not admitted"
        )
    is_replay = row.get("schema_version") == "general_coding_replay_v1"
    if is_replay and policy.get("type") != "geak_replay_release":
        raise DataContractError(
            f"{row.get('sample_id', '<unknown>')}: replay rows require "
            "geak_replay_release admission"
        )
    provenance = row.get("provenance")
    if policy.get("require_provenance", True) and (
        not isinstance(provenance, Mapping) or not provenance
    ):
        raise DataContractError(
            f"{row.get('sample_id', '<unknown>')}: provenance is required"
        )
    required_fields = policy.get("required_provenance_fields", [])
    if not isinstance(required_fields, list) or not all(
        isinstance(field, str) and field for field in required_fields
    ):
        raise DataContractError("required_provenance_fields must be strings")
    missing = [
        field
        for field in required_fields
        if not isinstance(provenance, Mapping) or not provenance.get(field)
    ]
    if missing:
        raise DataContractError(
            f"{row.get('sample_id', '<unknown>')}: missing provenance fields "
            + ", ".join(missing)
        )
