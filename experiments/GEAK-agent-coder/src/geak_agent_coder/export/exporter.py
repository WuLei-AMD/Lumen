"""Shard-wise HF blockwise-FP8 export with explicit deployment gating."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
from pathlib import Path
from typing import Mapping

from .fp8 import (
    BLOCK_SIZE,
    fp8_dtype_for_format,
    fp8_format_for_dtype,
    quantization_error_metrics,
    quantize_blockwise_2d,
    scale_key_for,
    weight_eligibility,
)

FORMAT = "geak_hf_blockwise2d_fp8_v1"
CANDIDATE_MARKER = "CANDIDATE"
DEPLOYABLE_MARKER = "DEPLOYABLE"


def quantize_state(
    state: Mapping[str, object],
    *,
    fp8_dtype=None,
    block_size: int = BLOCK_SIZE,
):
    output = {}
    quantized: list[str] = []
    skipped: dict[str, str] = {}
    validation: dict[str, dict] = {}
    for name, value in state.items():
        eligibility = weight_eligibility(name, value, block_size=block_size)
        if not eligibility.eligible:
            output[name] = value
            skipped[name] = eligibility.reason
            continue
        fp8_weight, scale_inv = quantize_blockwise_2d(
            value, fp8_dtype=fp8_dtype, block_size=block_size
        )
        output[name] = fp8_weight.cpu()
        output[scale_key_for(name)] = scale_inv.cpu()
        quantized.append(name)
        validation[name] = quantization_error_metrics(
            value, fp8_weight, scale_inv, block_size=block_size
        )
    return output, quantized, skipped, validation


def _artifact_digest(root: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        if not path.exists():
            continue
        digest.update(name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _model_shards(source: Path):
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return sorted(set(index["weight_map"].values())), index
    return ["model.safetensors"], None


def export_fp8_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    fp8_dtype=None,
    fp8_format: str | None = None,
    target_arch: str | None = None,
    block_size: int = BLOCK_SIZE,
) -> dict:
    """Quantize one HF shard at a time and mark output candidate-only."""
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("FP8 HF export requires safetensors") from exc

    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if fp8_dtype is None:
        if fp8_format is None:
            fp8_format = "e4m3fn"
        fp8_dtype = fp8_dtype_for_format(fp8_format)
    actual_fp8_format = fp8_format_for_dtype(fp8_dtype)
    if fp8_format is not None and fp8_format != actual_fp8_format:
        raise ValueError(
            f"declared fp8_format={fp8_format!r} does not match dtype "
            f"format={actual_fp8_format!r}"
        )

    source, output = Path(source_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shard_names, index = _model_shards(source)
    reports = []
    weight_map = dict(index["weight_map"]) if index else {}
    total_quantized = 0
    total_size = 0
    minimum_snr_db = float("inf")
    maximum_rmse = 0.0
    for shard_name in shard_names:
        state = load_file(str(source / shard_name), device="cpu")
        converted, quantized, skipped, validation = quantize_state(
            state, fp8_dtype=fp8_dtype, block_size=block_size
        )
        invalid = [
            name
            for name, metrics in validation.items()
            if not metrics["finite"] or metrics["snr_db"] < 8.0
        ]
        if invalid:
            raise RuntimeError(
                "FP8 numerical validation failed for weights: "
                + ", ".join(invalid[:8])
            )
        if validation:
            minimum_snr_db = min(
                minimum_snr_db,
                *(float(metrics["snr_db"]) for metrics in validation.values()),
            )
            maximum_rmse = max(
                maximum_rmse,
                *(float(metrics["rmse"]) for metrics in validation.values()),
            )
        for key in quantized:
            weight_map[scale_key_for(key)] = shard_name
        total_size += sum(
            value.numel() * value.element_size() for value in converted.values()
        )
        temporary = output / f".{shard_name}.tmp"
        save_file(converted, str(temporary))
        os.replace(temporary, output / shard_name)
        reports.append(
            {
                "file": shard_name,
                "quantized_weights": len(quantized),
                "preserved_weights": len(skipped),
                "minimum_snr_db": min(
                    (float(metrics["snr_db"]) for metrics in validation.values()),
                    default=None,
                ),
            }
        )
        total_quantized += len(quantized)
        del state, converted
    if total_quantized == 0:
        raise RuntimeError("eligibility policy selected no FP8 weights")

    if index:
        output_index = dict(index)
        output_index["weight_map"] = weight_map
        output_index["metadata"] = dict(output_index.get("metadata", {}))
        output_index["metadata"]["total_size"] = total_size
        (output / "model.safetensors.index.json").write_text(
            json.dumps(output_index, indent=2) + "\n", encoding="utf-8"
        )
    for path in source.iterdir():
        if path.is_file() and path.name not in {
            *shard_names,
            "model.safetensors.index.json",
            CANDIDATE_MARKER,
            DEPLOYABLE_MARKER,
        }:
            shutil.copy2(path, output / path.name)

    quantization_config = {
        "quant_method": "fp8",
        "fmt": actual_fp8_format,
        "activation_scheme": "dynamic",
        "weight_block_size": [block_size, block_size],
    }
    config_path = output / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["quantization_config"] = quantization_config
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    identity_files = [*shard_names, "config.json"]
    if index:
        identity_files.append("model.safetensors.index.json")
    artifact_sha256 = _artifact_digest(output, identity_files)
    manifest = {
        "format": FORMAT,
        "target_arch": target_arch or "unspecified",
        "fp8_format": actual_fp8_format,
        "block_size": [block_size, block_size],
        "scale_key": "weight_scale_inv",
        "scale_semantics": "dequantization_multiplier",
        "quantization_config": quantization_config,
        "source": str(source),
        "shards": reports,
        "quantized_weights": total_quantized,
        "minimum_snr_db": minimum_snr_db,
        "maximum_rmse": maximum_rmse,
        "artifact_sha256": artifact_sha256,
        "deployment_status": "candidate",
        "vllm_smoke_validated": False,
    }
    (output / "fp8_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / CANDIDATE_MARKER).write_text(
        "Candidate artifact; no vLLM deployability claim.\n", encoding="utf-8"
    )
    (output / DEPLOYABLE_MARKER).unlink(missing_ok=True)
    return manifest


def mark_deployable(output_dir: str | Path, smoke_evidence: str | Path) -> dict:
    """Promote only explicit, successful vLLM smoke-test evidence."""
    output = Path(output_dir)
    manifest_path = output / "fp8_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = json.loads(Path(smoke_evidence).read_text(encoding="utf-8"))
    artifact_sha256 = manifest.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
        raise ValueError("candidate manifest has no pinned artifact_sha256")
    if manifest.get("target_arch") in {None, "", "unspecified"}:
        raise ValueError("candidate manifest has no deployable target_arch")
    if manifest.get("fp8_format") not in {"e4m3fn", "e4m3fnuz"}:
        raise ValueError("candidate manifest has no supported fp8_format")
    required = {
        "engine": "vllm",
        "status": "passed",
        "artifact_format": FORMAT,
        "artifact_sha256": artifact_sha256,
    }
    if "target_arch" in manifest:
        required["device_arch"] = manifest["target_arch"]
    if "fp8_format" in manifest:
        required["fp8_format"] = manifest["fp8_format"]
    mismatches = {
        key: (evidence.get(key), expected)
        for key, expected in required.items()
        if evidence.get(key) != expected
    }
    if mismatches or not evidence.get("command") or not evidence.get("timestamp"):
        raise ValueError(
            "deployable gate requires passed vLLM smoke evidence with "
            f"command/timestamp; mismatches={mismatches}"
        )
    manifest["deployment_status"] = "deployable"
    manifest["vllm_smoke_validated"] = True
    manifest["smoke_evidence"] = Path(smoke_evidence).name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    evidence_target = output / Path(smoke_evidence).name
    if Path(smoke_evidence).resolve() != evidence_target.resolve():
        shutil.copy2(smoke_evidence, evidence_target)
    (output / CANDIDATE_MARKER).unlink(missing_ok=True)
    (output / DEPLOYABLE_MARKER).write_text(
        "Deployable gate passed with recorded vLLM smoke evidence.\n", encoding="utf-8"
    )
    return manifest
