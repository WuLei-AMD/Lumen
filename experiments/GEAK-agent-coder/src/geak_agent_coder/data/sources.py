"""Closed registry of local-only raw dataset sources."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .contracts import DataContractError

SourceLoader = Callable[[Mapping[str, Any]], Iterator[dict[str, Any]]]
_SOURCES: dict[str, SourceLoader] = {}


def register_source(name: str, loader: SourceLoader) -> None:
    if not isinstance(name, str) or not name.replace("_", "").isalnum():
        raise ValueError("source name must be a simple identifier")
    if name in _SOURCES:
        raise ValueError(f"source already registered: {name}")
    if not callable(loader):
        raise TypeError("source loader must be callable")
    _SOURCES[name] = loader


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise DataContractError(f"local JSONL file does not exist: {path}")
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
            if not isinstance(row, Mapping):
                raise DataContractError(f"{path}:{line_number}: row must be an object")
            yield dict(row)


def _local_jsonl(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    path = config.get("path")
    if not isinstance(path, str):
        raise DataContractError("local_jsonl source requires path")
    yield from _jsonl(Path(path).expanduser().resolve())


def _iter_dataset(dataset: Any) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(dataset):
        if not isinstance(row, Mapping):
            raise DataContractError(f"dataset row {index} must be an object")
        yield dict(row)


def _local_hf_layout(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    path_value = config.get("path")
    if not isinstance(path_value, str):
        raise DataContractError("local_hf_layout source requires path")
    path = Path(path_value).expanduser().resolve()
    split = config.get("split")
    if (path / "samples.jsonl").is_file():
        yield from _jsonl(path / "samples.jsonl")
        return
    split_name = str(split or "train")
    for candidate in (
        path / f"{split_name}.jsonl",
        path / "data" / f"{split_name}.jsonl",
    ):
        if candidate.is_file():
            yield from _jsonl(candidate)
            return
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise DataContractError(
            "reading an Arrow HF layout requires the datasets package"
        ) from exc
    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        if split_name not in loaded:
            raise DataContractError(f"HF layout has no split {split_name!r}")
        loaded = loaded[split_name]
    yield from _iter_dataset(loaded)


def _hf_cache(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    dataset_id = config.get("dataset_id")
    cache_dir = config.get("cache_dir")
    split = config.get("split", "train")
    revision = config.get("revision")
    if not all(
        isinstance(value, str) and value
        for value in (dataset_id, cache_dir, split, revision)
    ):
        raise DataContractError(
            "hf_cache requires dataset_id, pinned revision, cache_dir, and split"
        )
    allow_network = config.get("allow_network", False)
    if not isinstance(allow_network, bool):
        raise DataContractError("allow_network must be boolean")
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise DataContractError("hf_cache requires the datasets package") from exc
    previous = os.environ.get("HF_DATASETS_OFFLINE")
    if not allow_network:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        loaded = load_dataset(
            dataset_id,
            name=config.get("name"),
            split=split,
            revision=revision,
            cache_dir=str(Path(cache_dir).expanduser().resolve()),
            download_config=DownloadConfig(local_files_only=not allow_network),
            download_mode="reuse_dataset_if_exists",
        )
    except Exception as exc:
        raise DataContractError(
            f"cached HF dataset unavailable without network: {dataset_id}"
        ) from exc
    finally:
        if previous is None:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
        else:
            os.environ["HF_DATASETS_OFFLINE"] = previous
    yield from _iter_dataset(loaded)


def iter_source(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    source_type = config.get("type")
    if not isinstance(source_type, str) or source_type not in _SOURCES:
        raise DataContractError(f"unknown source type: {source_type!r}")
    yield from _SOURCES[source_type](config)


register_source("local_jsonl", _local_jsonl)
register_source("local_hf_layout", _local_hf_layout)
register_source("hf_cache", _hf_cache)
