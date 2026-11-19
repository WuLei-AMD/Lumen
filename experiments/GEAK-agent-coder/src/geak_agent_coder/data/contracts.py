"""Dataset loading, schema validation, and release gates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

SUPPORTED_SCHEMAS = frozenset(
    {
        "geak_kernel_sft_v1",
        "general_coding_replay_v1",
        "generic_coding_sft_v1",
    }
)
TRAIN_SPLIT = "train"
FORBIDDEN_TRAIN_SPLITS = frozenset({"dev", "held_out", "held-out", "test"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DataContractError(ValueError):
    """Raised when data cannot safely enter training."""


class SchemaValidationError(DataContractError):
    """Raised when a sample does not satisfy its declared schema."""


class TrainingSplitError(DataContractError):
    """Raised when evaluation data is presented to a training loader."""


class ReplayGateError(DataContractError):
    """Raised when the final replay package is not training-ready."""


def _require(
    condition: bool, sample_id: str, field: str, detail: str
) -> None:
    if not condition:
        raise SchemaValidationError(f"{sample_id}: {field} {detail}")


def _mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def validate_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached sample mapping.

    Validation is intentionally limited to the stable training contract. Deep
    artifact verification belongs to the dataset production pipeline.
    """

    _require(_mapping(sample), "<unknown>", "sample", "must be an object")
    result = dict(sample)
    sample_id = str(result.get("sample_id") or "<unknown>")
    schema = result.get("schema_version")
    _require(
        schema in SUPPORTED_SCHEMAS,
        sample_id,
        "schema_version",
        f"must be one of {sorted(SUPPORTED_SCHEMAS)}",
    )
    _require(
        isinstance(result.get("sample_id"), str)
        and bool(result["sample_id"].strip()),
        sample_id,
        "sample_id",
        "must be a non-empty string",
    )
    split = result.get("split")
    _require(
        isinstance(split, str) and bool(split.strip()),
        sample_id,
        "split",
        "must be a non-empty string",
    )
    _require(
        isinstance(result.get("task_type"), str)
        and bool(result["task_type"].strip()),
        sample_id,
        "task_type",
        "must be a non-empty string",
    )
    _require(_mapping(result.get("input")), sample_id, "input", "must be an object")
    output = result.get("output")
    _require(_mapping(output), sample_id, "output", "must be an object")
    response = output.get("patch")
    if schema == "generic_coding_sft_v1":
        response = output.get("response", response)
    _require(
        isinstance(response, str) and bool(response),
        sample_id,
        "output.patch/response",
        "must contain a non-empty assistant response",
    )
    _require(
        _mapping(result.get("provenance")),
        sample_id,
        "provenance",
        "must be an object",
    )

    if schema == "geak_kernel_sft_v1":
        _require(
            result.get("sample_domain", "kernel") == "kernel",
            sample_id,
            "sample_domain",
            "must be kernel",
        )
        labels = result.get("labels")
        _require(_mapping(labels), sample_id, "labels", "must be an object")
        for field in (
            "patch_applies",
            "compile_pass",
            "correctness_pass",
            "benchmark_valid",
        ):
            _require(
                isinstance(labels.get(field), bool),
                sample_id,
                f"labels.{field}",
                "must be boolean",
            )
    elif schema == "general_coding_replay_v1":
        _require(
            result.get("sample_domain") == "general_coding",
            sample_id,
            "sample_domain",
            "must be general_coding",
        )
        _require(
            result.get("task_type") == "general_coding_replay",
            sample_id,
            "task_type",
            "must be general_coding_replay",
        )
        _require(
            isinstance(result.get("coding_task_type"), str)
            and bool(result["coding_task_type"].strip()),
            sample_id,
            "coding_task_type",
            "must be a non-empty string",
        )
    else:
        _require(
            result.get("sample_domain") == "generic_coding",
            sample_id,
            "sample_domain",
            "must be generic_coding",
        )
    return result


def iter_local_samples(source: str | Path | Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Load a local HF-layout ``samples.jsonl`` or dataset-like iterable.

    No datasets/transformers import occurs. A ``datasets.Dataset`` works via
    its normal iteration protocol and therefore never triggers a network load.
    """

    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.is_dir():
            path = path / "samples.jsonl"
        if not path.is_file():
            raise DataContractError(f"local samples file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataContractError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not _mapping(row):
                    raise DataContractError(
                        f"{path}:{line_number}: sample must be an object"
                    )
                yield validate_sample(row)
        return

    try:
        iterator = iter(source)
    except TypeError as exc:
        raise DataContractError(
            "source must be a local path or an iterable dataset"
        ) from exc
    for index, row in enumerate(iterator):
        if not _mapping(row):
            raise DataContractError(f"dataset row {index} must be an object")
        yield validate_sample(row)


def load_samples(
    source: str | Path | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize validated samples without applying a split policy."""

    return list(iter_local_samples(source))


def load_training_samples(
    source: str | Path | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Load training rows, failing if *any* non-training split is present."""

    rows = load_samples(source)
    invalid = [
        f"{row['sample_id']}:{row['split']}"
        for row in rows
        if str(row["split"]).lower() != TRAIN_SPLIT
    ]
    if invalid:
        raise TrainingSplitError(
            "training input contains non-train rows (filter explicitly before "
            f"loading): {', '.join(invalid[:10])}"
        )
    return rows


@dataclass(frozen=True)
class ReplayGateResult:
    accepted_count: int
    leakage_passed: bool
    package_ready: bool
    trust_passed: bool
    checksums_verified: int
    required_accepted_count: int = 500

    @property
    def ready(self) -> bool:
        return (
            self.accepted_count == self.required_accepted_count
            and self.leakage_passed
            and self.package_ready
            and self.trust_passed
            and self.checksums_verified > 0
        )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayGateError(f"cannot read gate artifact {path}: {exc}") from exc


def _accepted_count(path: Path) -> int:
    if not path.is_file():
        raise ReplayGateError(f"accepted replay file missing: {path}")
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayGateError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, Mapping):
                raise ReplayGateError(f"{path}:{line_number}: row is not an object")
            count += 1
    return count


def _report_passed(report: Any, *, kind: str) -> bool:
    if not isinstance(report, Mapping):
        return False
    for key in ("passed", "pass", "ready"):
        if isinstance(report.get(key), bool):
            return bool(report[key])
    status = str(report.get("status", "")).lower()
    if status in {"fail", "failed", "blocked", "error"}:
        return False
    if status in {"pass", "passed", "ready", "complete"}:
        return True
    if kind == "leakage":
        counts = [
            value
            for key, value in report.items()
            if "overlap" in str(key).lower() and isinstance(value, int)
        ]
        return bool(counts) and all(value == 0 for value in counts)
    return False


def _verify_checksums(root: Path, checksum_path: Path) -> int:
    if not checksum_path.is_file():
        raise ReplayGateError(f"checksum file missing: {checksum_path}")
    verified = 0
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256_RE.fullmatch(parts[0]):
            raise ReplayGateError(
                f"{checksum_path}:{line_number}: invalid checksum entry"
            )
        relative = parts[1].lstrip("*")
        target = (root / relative).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise ReplayGateError(
                f"{checksum_path}:{line_number}: path escapes package"
            ) from exc
        if not target.is_file():
            raise ReplayGateError(f"checksummed file missing: {relative}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != parts[0]:
            raise ReplayGateError(f"checksum mismatch: {relative}")
        verified += 1
    if verified == 0:
        raise ReplayGateError("checksums.sha256 contains no entries")
    return verified


def enforce_final_replay_gates(
    package_root: str | Path,
    *,
    required_accepted_count: int = 500,
    accepted_file: str = "accepted.jsonl",
    leakage_report: str = "leakage-report.json",
    package_manifest: str = "manifest.json",
    trust_report: str = "trust-report.json",
    checksum_file: str = "checksums.sha256",
) -> ReplayGateResult:
    """Require exact replay count, leakage/trust pass, package readiness, and hashes."""

    if required_accepted_count <= 0:
        raise ReplayGateError("required_accepted_count must be positive")
    root = Path(package_root).expanduser().resolve()
    count = _accepted_count(root / accepted_file)
    leakage_ok = _report_passed(
        _read_json(root / leakage_report), kind="leakage"
    )
    package_ok = _report_passed(
        _read_json(root / package_manifest), kind="package"
    )
    trust = _read_json(root / trust_report)
    trust_ok = (
        _report_passed(trust, kind="trust")
        and int(trust.get("offline_cpu_only_double_replay_passed", -1))
        == required_accepted_count
        and int(trust.get("quality_failures", -1)) == 0
    )
    verified = _verify_checksums(root, root / checksum_file)
    result = ReplayGateResult(
        count,
        leakage_ok,
        package_ok,
        trust_ok,
        verified,
        required_accepted_count,
    )
    failures = []
    if count != required_accepted_count:
        failures.append(
            f"accepted={count}, required={required_accepted_count}"
        )
    if not leakage_ok:
        failures.append("leakage report did not pass")
    if not package_ok:
        failures.append("package is not ready")
    if not trust_ok:
        failures.append(
            "trust report does not prove "
            f"{required_accepted_count} clean offline double replays"
        )
    if failures:
        raise ReplayGateError("; ".join(failures))
    return result
