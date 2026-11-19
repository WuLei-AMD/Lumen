"""Loader and structural validation for ``geak_data_config_v2``."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .contracts import DataContractError


def _resolve(base: Path, value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value).expanduser()
    return str((base / path).resolve()) if not path.is_absolute() else str(path.resolve())


def load_data_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("loading data config requires PyYAML") from exc
    config_path = Path(path).expanduser().resolve()
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DataContractError(f"cannot load data config {config_path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise DataContractError("data config must be an object")
    config = deepcopy(dict(loaded))
    if config.get("schema_version") != "geak_data_config_v2":
        raise DataContractError("schema_version must be geak_data_config_v2")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DataContractError("sources must be a non-empty list")
    names: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise DataContractError(f"sources[{index}] must be an object")
        name = source.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise DataContractError(f"sources[{index}].name must be unique")
        names.add(name)
        if not isinstance(source.get("field_mapping"), Mapping):
            raise DataContractError(f"sources[{index}].field_mapping is required")
        if not isinstance(source.get("admission", {}), Mapping):
            raise DataContractError(f"sources[{index}].admission must be an object")
        for key in ("path", "cache_dir"):
            if key in source:
                source[key] = _resolve(config_path.parent, source[key])
        admission = source.setdefault("admission", {})
        for key in ("package_root", "checksum_path"):
            if key in admission:
                admission[key] = _resolve(config_path.parent, admission[key])
    tokenizer = config.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise DataContractError("tokenizer must be an object")
    tokenizer["path"] = _resolve(config_path.parent, tokenizer.get("path"))
    if not isinstance(tokenizer.get("path"), str):
        raise DataContractError("tokenizer.path must pin a local directory")
    expected = tokenizer.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise DataContractError("tokenizer.sha256 must pin a 64-character SHA256")
    output = config.get("output")
    if not isinstance(output, dict):
        raise DataContractError("output must be an object")
    output["directory"] = _resolve(config_path.parent, output.get("directory"))
    if not isinstance(output.get("directory"), str):
        raise DataContractError("output.directory is required")
    sampling = config.get("sampling")
    if not isinstance(sampling, Mapping):
        raise DataContractError("sampling must be an object")
    steps = sampling.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise DataContractError("sampling.steps must be a positive integer")
    config["_config_path"] = str(config_path)
    return config
