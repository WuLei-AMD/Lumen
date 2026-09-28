"""End-to-end local data build orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .admission import admit_row, path_sha256, verify_source_admission
from .config import load_data_config
from .contracts import DataContractError, TrainingSplitError, validate_sample
from .formatting import TokenizationConfig, TokenizerLike, tokenize_samples
from .manifest import build_data_manifest, write_manifest, write_tokenized_jsonl
from .mapping import map_row
from .sampling import SamplingConfig, build_sampling_plan
from .sources import iter_source


def load_local_tokenizer(config: Mapping[str, Any]) -> TokenizerLike:
    path = Path(str(config["path"])).expanduser().resolve()
    if not path.is_dir():
        raise DataContractError(f"local tokenizer directory does not exist: {path}")
    actual = path_sha256(path)
    if actual != str(config["sha256"]).lower():
        raise DataContractError(
            f"tokenizer checksum mismatch: expected {config['sha256']}, got {actual}"
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("loading a tokenizer requires transformers") from exc
    return AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
    )


def _tokenization_config(config: Mapping[str, Any]) -> TokenizationConfig:
    raw = config.get("tokenization", {})
    if not isinstance(raw, Mapping):
        raise DataContractError("tokenization must be an object")
    return TokenizationConfig(
        model_family=str(raw.get("model_family", "qwen3-30b-a3b")),
        max_length=int(raw.get("max_length", 32768)),
        include_assistant_eot_in_loss=bool(
            raw.get("include_assistant_eot_in_loss", True)
        ),
        overflow_policy=str(raw.get("overflow_policy", "error")),
        expected_chat_template_sha256=raw.get("chat_template_sha256"),
    )


def _sampling_config(config: Mapping[str, Any]) -> SamplingConfig:
    raw = config["sampling"]
    mix = raw.get("replay_mix", {})
    if not isinstance(mix, Mapping):
        raise DataContractError("sampling.replay_mix must be an object")
    return SamplingConfig(
        replay_min_share=float(mix.get("min_assistant_loss_token_share", 0.15)),
        replay_max_share=float(mix.get("max_assistant_loss_token_share", 0.20)),
        replay_mix_enabled=bool(mix.get("enabled", True)),
        seed=int(raw.get("seed", 0)),
        length_penalty_power=float(raw.get("length_penalty_power", 0.5)),
    )


def _immutable_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
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
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return
        raise DataContractError(f"immutable manifest already exists with other content: {path}")
    write_manifest(path, manifest)
    path.chmod(0o444)


def _output_path(directory: Path, name: Any, default: str) -> Path:
    if name is None:
        name = default
    if not isinstance(name, str) or not name:
        raise DataContractError("output filenames must be non-empty strings")
    path = (directory / name).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as exc:
        raise DataContractError(f"output path escapes directory: {name}") from exc
    return path


def build_data(
    config: Mapping[str, Any],
    tokenizer: TokenizerLike,
) -> dict[str, Any]:
    """Normalize, admit, tokenize, schedule, and persist one pinned build."""

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_manifest: list[dict[str, Any]] = []
    for source in config["sources"]:
        policy = source.get("admission", {})
        identity = verify_source_admission(source, policy)
        row_count = 0
        for raw in iter_source(source):
            row = validate_sample(map_row(raw, source["field_mapping"]))
            admit_row(row, policy)
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise DataContractError(f"duplicate sample_id: {sample_id}")
            seen.add(sample_id)
            split = str(row["split"]).lower()
            if split == "train":
                train_rows.append(row)
            elif split == "dev":
                dev_rows.append(row)
            else:
                raise TrainingSplitError(
                    f"{sample_id}: build only materializes train and dev, got {split}"
                )
            row_count += 1
        source_manifest.append(
            {"name": source["name"], "type": source["type"], "rows": row_count, **identity}
        )
    if not train_rows:
        raise DataContractError("build has no admitted training rows")

    token_config = _tokenization_config(config)
    tokenized_train, quarantined_train = tokenize_samples(
        train_rows, tokenizer, token_config
    )
    tokenized_dev, quarantined_dev = tokenize_samples(dev_rows, tokenizer, token_config)
    if not tokenized_train:
        raise DataContractError("all training rows were quarantined")
    sampling_config = _sampling_config(config)
    plan = build_sampling_plan(
        tokenized_train,
        steps=int(config["sampling"]["steps"]),
        config=sampling_config,
    )
    scheduled = [tokenized_train[index] for index in plan.indices]

    output = config["output"]
    directory = Path(output["directory"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    train_path = _output_path(
        directory, output.get("train_file"), "train.tokenized.jsonl"
    )
    dev_path = _output_path(
        directory, output.get("dev_file"), "dev.tokenized.jsonl"
    )
    manifest_path = _output_path(
        directory, output.get("manifest_file"), "manifest.json"
    )
    if manifest_path.exists():
        raise DataContractError(f"immutable manifest already exists: {manifest_path}")
    write_tokenized_jsonl(train_path, scheduled)
    write_tokenized_jsonl(dev_path, tokenized_dev)

    token_dict = asdict(token_config)
    manifest = build_data_manifest(
        train_rows,
        tokenized_train,
        tokenization_config=token_dict,
        quarantined=quarantined_train,
        sampling_plan=plan,
    )
    config_path = config.get("_config_path")
    manifest.update(
        {
            "schema_version": "geak_training_data_manifest_v2",
            "sources": source_manifest,
            "tokenizer": {
                "path": str(config["tokenizer"]["path"]),
                "sha256": str(config["tokenizer"]["sha256"]).lower(),
            },
            "sampling_config": asdict(sampling_config),
            "dev": {
                "source_rows": len(dev_rows),
                "tokenized_rows": len(tokenized_dev),
                "quarantined": [item.as_dict() for item in quarantined_dev],
            },
            "artifacts": {
                "train": {
                    "path": train_path.name,
                    "rows": len(scheduled),
                    "sha256": path_sha256(train_path),
                },
                "dev": {
                    "path": dev_path.name,
                    "rows": len(tokenized_dev),
                    "sha256": path_sha256(dev_path),
                },
            },
            "config_sha256": (
                hashlib.sha256(Path(str(config_path)).read_bytes()).hexdigest()
                if isinstance(config_path, str)
                else None
            ),
        }
    )
    _immutable_manifest(manifest_path, manifest)
    return manifest


def build_from_config(
    path: str | Path, *, tokenizer: TokenizerLike | None = None
) -> dict[str, Any]:
    config = load_data_config(path)
    if tokenizer is None:
        tokenizer = load_local_tokenizer(config["tokenizer"])
    else:
        actual = path_sha256(config["tokenizer"]["path"])
        if actual != str(config["tokenizer"]["sha256"]).lower():
            raise DataContractError(
                f"tokenizer checksum mismatch: expected {config['tokenizer']['sha256']}, "
                f"got {actual}"
            )
    return build_data(config, tokenizer)
