"""Adapter-only checkpoint helpers with an FSDP-compatible gathering seam."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Mapping

ADAPTER_FILE = "adapter_model.safetensors"
METADATA_FILE = "adapter_config.json"


def is_adapter_key(name: str) -> bool:
    return name.endswith(".lora_A") or name.endswith(".lora_B")


def _full_cpu_tensor(tensor):
    """Materialize a Tensor/DTensor; all ranks must call this for DTensors."""
    full_tensor = getattr(tensor, "full_tensor", None)
    if callable(full_tensor):
        tensor = full_tensor()
    return tensor.detach().to(device="cpu").contiguous()


def extract_adapter_state(
    model,
    *,
    state_dict_getter: Callable[[object], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Extract adapter tensors.

    ``state_dict_getter`` is the interface for FSDP1/FSDP2 callers that need a
    specific full-state context. Without it, DTensor ``full_tensor()`` provides
    the FSDP2-aware path and must be entered collectively on every rank.
    """
    source = (
        state_dict_getter(model)
        if state_dict_getter is not None
        else dict(model.named_parameters())
    )
    state = {name: _full_cpu_tensor(value) for name, value in source.items() if is_adapter_key(name)}
    if not state:
        raise RuntimeError("no adapter tensors found")
    return state


def adapter_metadata(
    model,
    *,
    base_model: str,
    revision: str | None = None,
    world_size: int = 1,
    tokenizer_sha256: str | None = None,
    data_manifest_sha256: str | None = None,
) -> dict:
    modules = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            modules[name] = {
                "rank": int(module.rank),
                "alpha": float(module.alpha),
                "dtype": str(module.lora_A.dtype).removeprefix("torch."),
            }
    expert_layouts = {
        (int(module.experts_per_rank), int(module.num_experts))
        for module in model.modules()
        if hasattr(module, "experts_per_rank") and hasattr(module, "num_experts")
    }
    metadata = {
        "format": "geak_hierarchical_lora_v1",
        "base_model": base_model,
        "base_model_revision": revision,
        "world_size": world_size,
        "tokenizer_sha256": tokenizer_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "modules": modules,
    }
    if expert_layouts:
        if len(expert_layouts) != 1:
            raise RuntimeError(f"inconsistent expert layouts: {sorted(expert_layouts)}")
        experts_per_rank, num_experts = expert_layouts.pop()
        metadata["expert_parallel_size"] = num_experts // experts_per_rank
        metadata["experts_per_rank"] = experts_per_rank
        metadata["num_experts"] = num_experts
    return metadata


def save_adapter(
    model,
    output_dir: str | Path,
    *,
    base_model: str,
    revision: str | None = None,
    rank: int = 0,
    world_size: int = 1,
    state_dict_getter: Callable[[object], Mapping[str, object]] | None = None,
    tokenizer_sha256: str | None = None,
    data_manifest_sha256: str | None = None,
) -> dict:
    """Collect adapters and atomically write one EP-local shard per rank."""
    state = extract_adapter_state(model, state_dict_getter=state_dict_getter)
    if world_size > 1 and rank != 0:
        state = {
            name: value
            for name, value in state.items()
            if ".local_experts." in f".{name}."
        }
    metadata = adapter_metadata(
        model,
        base_model=base_model,
        revision=revision,
        world_size=world_size,
        tokenizer_sha256=tokenizer_sha256,
        data_manifest_sha256=data_manifest_sha256,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("saving adapters requires safetensors") from exc
    adapter_name = (
        ADAPTER_FILE
        if world_size == 1
        else f"adapter_model.rank{rank:05d}.safetensors"
    )
    temporary = output / f".{adapter_name}.tmp"
    save_file(state, str(temporary))
    os.replace(temporary, output / adapter_name)
    if rank == 0:
        config_temp = output / f".{METADATA_FILE}.tmp"
        config_temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(config_temp, output / METADATA_FILE)
    return metadata


def load_adapter_state(path: str | Path, *, device: str = "cpu") -> tuple[dict, dict]:
    path = Path(path)
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("loading adapters requires safetensors") from exc
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    if metadata.get("world_size", 1) != 1:
        raise RuntimeError("load_adapter only accepts a single-rank adapter; use merge utility")
    state = load_file(str(path / ADAPTER_FILE), device=device)
    return state, metadata


def load_adapter(model, path: str | Path, *, strict: bool = True) -> dict:
    state, metadata = load_adapter_state(path)
    expected = {name for name, _ in model.named_parameters() if is_adapter_key(name)}
    supplied = set(state)
    if strict and expected != supplied:
        raise RuntimeError(
            f"adapter keys differ: missing={sorted(expected - supplied)}, "
            f"unexpected={sorted(supplied - expected)}"
        )
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if name in parameters:
            parameters[name].data.copy_(value.to(parameters[name].device, parameters[name].dtype))
    return metadata


def load_distributed_adapter(
    model,
    path: str | Path,
    *,
    rank: int,
    world_size: int,
    strict: bool = True,
    expected_base_model: str | None = None,
    expected_revision: str | None = None,
    expected_tokenizer_sha256: str | None = None,
    expected_data_manifest_sha256: str | None = None,
) -> dict:
    """Load shared adapters from rank 0 and this EP rank's local experts."""

    path = Path(path)
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("loading adapters requires safetensors") from exc
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    if int(metadata.get("world_size", -1)) != world_size:
        raise RuntimeError(
            "adapter world size mismatch: "
            f"checkpoint={metadata.get('world_size')} runtime={world_size}"
        )
    expected_identity = {
        "base_model": expected_base_model,
        "base_model_revision": expected_revision,
        "tokenizer_sha256": expected_tokenizer_sha256,
        "data_manifest_sha256": expected_data_manifest_sha256,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_identity.items()
        if expected is not None and metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"adapter identity mismatch: {mismatches}")
    rank_zero = load_file(
        str(path / "adapter_model.rank00000.safetensors"), device="cpu"
    )
    state = {
        name: value
        for name, value in rank_zero.items()
        if ".local_experts." not in f".{name}."
    }
    local = load_file(
        str(path / f"adapter_model.rank{rank:05d}.safetensors"), device="cpu"
    )
    state.update(
        {
            name: value
            for name, value in local.items()
            if ".local_experts." in f".{name}."
        }
    )
    expected = {name for name, _ in model.named_parameters() if is_adapter_key(name)}
    supplied = set(state)
    if strict and expected != supplied:
        raise RuntimeError(
            f"adapter keys differ on rank {rank}: "
            f"missing={sorted(expected - supplied)}, "
            f"unexpected={sorted(supplied - expected)}"
        )
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if name in parameters:
            parameters[name].data.copy_(
                value.to(parameters[name].device, parameters[name].dtype)
            )
    return metadata
