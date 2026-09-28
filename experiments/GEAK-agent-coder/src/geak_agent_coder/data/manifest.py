"""Aggregate statistics and deterministic data manifest output."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .formatting import QuarantinedSample, TokenizedSample
from .sampling import SamplingPlan


def _counts(samples: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(sample.get(field, "<missing>")) for sample in samples).items()
        )
    )


def aggregate_stats(
    samples: Sequence[Mapping[str, Any]],
    tokenized: Sequence[TokenizedSample],
    quarantined: Sequence[QuarantinedSample] = (),
) -> dict[str, Any]:
    """Compute auditable row and assistant-token statistics."""

    by_id = {sample.sample_id: sample for sample in tokenized}
    domain_loss: Counter[str] = Counter()
    domain_rows: Counter[str] = Counter()
    lengths: list[int] = []
    for sample in tokenized:
        domain_loss[sample.sample_domain] += sample.assistant_loss_tokens
        domain_rows[sample.sample_domain] += 1
        lengths.append(len(sample.input_ids))
    return {
        "source_rows": len(samples),
        "tokenized_rows": len(tokenized),
        "quarantined_rows": len(quarantined),
        "schema_counts": _counts(samples, "schema_version"),
        "split_counts": _counts(samples, "split"),
        "domain_counts": _counts(samples, "sample_domain"),
        "task_type_counts": _counts(samples, "task_type"),
        "tokenized_domain_rows": dict(sorted(domain_rows.items())),
        "assistant_loss_tokens": dict(sorted(domain_loss.items())),
        "total_assistant_loss_tokens": sum(domain_loss.values()),
        "min_sequence_tokens": min(lengths, default=0),
        "max_sequence_tokens": max(lengths, default=0),
        "quarantined_sample_ids": sorted(item.sample_id for item in quarantined),
        "missing_tokenized_sample_ids": sorted(
            str(sample["sample_id"])
            for sample in samples
            if str(sample["sample_id"]) not in by_id
            and str(sample["sample_id"])
            not in {item.sample_id for item in quarantined}
        ),
    }


def build_data_manifest(
    samples: Sequence[Mapping[str, Any]],
    tokenized: Sequence[TokenizedSample],
    *,
    tokenization_config: Mapping[str, Any],
    quarantined: Sequence[QuarantinedSample] = (),
    sampling_plan: SamplingPlan | None = None,
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable training data manifest."""

    canonical_samples = sorted(
        (
            str(sample["sample_id"]),
            json.dumps(
                sample,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
        for sample in samples
    )
    identity = hashlib.sha256(
        "\n".join(payload for _, payload in canonical_samples).encode("utf-8")
    ).hexdigest()
    result: dict[str, Any] = {
        "schema_version": "geak_training_data_manifest_v1",
        "dataset_identity_sha256": identity,
        "tokenization": dict(sorted(tokenization_config.items())),
        "stats": aggregate_stats(samples, tokenized, quarantined),
        "quarantine": [item.as_dict() for item in sorted(
            quarantined, key=lambda item: item.sample_id
        )],
    }
    if sampling_plan is not None:
        result["sampling"] = sampling_plan.as_dict()
    return result


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Atomically write canonical JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_tokenized_jsonl(
    path: str | Path, samples: Sequence[TokenizedSample]
) -> None:
    """Atomically persist the pre-tokenized training contract."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            sample.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for sample in samples
    )
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
