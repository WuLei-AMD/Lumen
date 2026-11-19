"""Campaign configuration, hashing, and artifact manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import FixedBenchmarkIdentity, ModelVariant


@dataclass(frozen=True)
class Campaign:
    name: str
    fixed_identity: FixedBenchmarkIdentity
    variants: tuple[ModelVariant, ...]
    tasks_manifest: Path
    output_root: Path
    protected_sha256: tuple[str, ...]
    runner: Mapping[str, Any]
    gpu_isolation: Mapping[str, Any]


def load_campaign(path: str | Path) -> Campaign:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("campaign config must be an object")
    variants_raw = value.get("variants")
    if not isinstance(variants_raw, list) or len(variants_raw) < 2:
        raise ValueError("campaign requires at least two model variants")
    allowed_variant_keys = {"name", "base_url", "model", "checkpoint_identity"}
    variants = []
    for raw in variants_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("each variant must be an object")
        extras = set(raw) - allowed_variant_keys
        if extras:
            raise ValueError(
                "model variants may differ only by base_url/model/checkpoint identity; "
                f"unexpected keys: {sorted(extras)}"
            )
        variants.append(ModelVariant.from_mapping(raw))
    protected = value.get("protected_sha256")
    if not isinstance(protected, list) or not protected:
        raise ValueError("protected_sha256 must be a non-empty list")
    runner = value.get("runner")
    if not isinstance(runner, Mapping):
        raise ValueError("runner must be an object")
    if runner.get("role", "engineer") != "engineer":
        raise ValueError("primary benchmark runner is fixed to engineer-only")
    return Campaign(
        name=str(value.get("name") or config_path.stem),
        fixed_identity=FixedBenchmarkIdentity.from_mapping(value["fixed_identity"]),
        variants=tuple(variants),
        tasks_manifest=_relative_path(config_path, value, "tasks_manifest"),
        output_root=_relative_path(config_path, value, "output_root"),
        protected_sha256=tuple(str(item).lower() for item in protected),
        runner=dict(runner),
        gpu_isolation=dict(value.get("gpu_isolation") or {}),
    )


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_campaign_manifest(
    campaign: Campaign,
    *,
    status: str,
    artifacts: Mapping[str, Any],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": "geak_benchmark_campaign_v1",
        "name": campaign.name,
        "status": status,
        "fixed_identity": campaign.fixed_identity.to_dict(),
        "variants": [variant.public_dict() for variant in campaign.variants],
        "tasks_manifest": str(campaign.tasks_manifest),
        "protected_sha256": list(campaign.protected_sha256),
        "runner": dict(campaign.runner),
        "gpu_isolation": dict(campaign.gpu_isolation),
        "artifacts": dict(artifacts),
        "config_hash": canonical_hash(
            {
                "fixed_identity": campaign.fixed_identity.to_dict(),
                "runner": campaign.runner,
                "gpu_isolation": campaign.gpu_isolation,
            }
        ),
    }
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _relative_path(
    config_path: Path, value: Mapping[str, Any], key: str
) -> Path:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty path")
    path = Path(raw).expanduser()
    return (config_path.parent / path).resolve() if not path.is_absolute() else path
