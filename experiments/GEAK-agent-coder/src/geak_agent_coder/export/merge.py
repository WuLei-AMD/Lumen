"""Streaming-ish merge of EP-sharded GEAK LoRA into HF safetensor shards."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Mapping

_EXPERT_PATH = re.compile(
    r"^(?P<prefix>.+\.mlp)\.local_experts\.experts\.(?P<local>\d+)"
    r"\.(?P<projection>gate_up_proj|down_proj)$"
)


def _module_updates(adapter_state: Mapping[str, object]) -> dict[str, tuple[object, object]]:
    pairs: dict[str, dict[str, object]] = {}
    for name, tensor in adapter_state.items():
        if name.endswith(".lora_A"):
            pairs.setdefault(name[: -len(".lora_A")], {})["A"] = tensor
        elif name.endswith(".lora_B"):
            pairs.setdefault(name[: -len(".lora_B")], {})["B"] = tensor
    incomplete = [name for name, pair in pairs.items() if set(pair) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"incomplete LoRA pairs: {incomplete}")
    return {name: (pair["A"], pair["B"]) for name, pair in pairs.items()}


def adapter_updates(
    adapter_state: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    rank: int = 0,
) -> dict[str, list[tuple[int | None, object]]]:
    """Map adapter pairs to HF weight keys and optional packed-expert slices."""
    updates: dict[str, list[tuple[int | None, object]]] = {}
    ep_size = int(metadata.get("expert_parallel_size", 1))
    experts_per_rank = int(metadata.get("experts_per_rank", 0))
    ep_rank = rank % ep_size
    for module_name, (lora_a, lora_b) in _module_updates(adapter_state).items():
        module_metadata = metadata.get("modules", {}).get(module_name)
        if module_metadata is None:
            raise KeyError(f"adapter metadata is missing module {module_name}")
        delta = (lora_b.float() @ lora_a.float()) * (
            float(module_metadata["alpha"]) / int(module_metadata["rank"])
        )
        expert = _EXPERT_PATH.match(module_name)
        if expert:
            if not experts_per_rank:
                raise ValueError("expert adapter metadata has no experts_per_rank")
            global_expert = ep_rank * experts_per_rank + int(expert.group("local"))
            key = (
                f"{expert.group('prefix')}.experts."
                f"{expert.group('projection')}"
            )
            updates.setdefault(key, []).append((global_expert, delta))
        else:
            updates.setdefault(f"{module_name}.weight", []).append((None, delta))
    return updates


def merge_updates_into_state(
    state: Mapping[str, object],
    updates: Mapping[str, list[tuple[int | None, object]]],
) -> dict[str, object]:
    output = dict(state)
    for key in set(state).intersection(updates):
        weight = state[key]
        merged = weight.float().clone()
        for expert_index, delta in updates[key]:
            if expert_index is None:
                if tuple(delta.shape) != tuple(merged.shape):
                    raise ValueError(f"LoRA delta shape mismatch for {key}")
                merged.add_(delta)
            else:
                if merged.ndim != 3 or tuple(delta.shape) != tuple(merged[expert_index].shape):
                    raise ValueError(f"packed expert LoRA delta shape mismatch for {key}")
                merged[expert_index].add_(delta)
        output[key] = merged.to(weight.dtype)
    return output


def load_all_adapter_updates(adapter_dir: str | Path):
    """Load one adapter shard at a time and retain only BF16-sized deltas."""
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("merging adapters requires safetensors") from exc

    adapter_dir = Path(adapter_dir)
    metadata = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    paths = sorted(adapter_dir.glob("adapter_model.rank*.safetensors"))
    if not paths:
        paths = [adapter_dir / "adapter_model.safetensors"]
    combined: dict[str, list[tuple[int | None, object]]] = {}
    for rank, path in enumerate(paths):
        state = load_file(str(path), device="cpu")
        for key, values in adapter_updates(state, metadata, rank=rank).items():
            combined.setdefault(key, []).extend(values)
    return combined, metadata


def merge_hf_checkpoint(
    base_dir: str | Path,
    adapter_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Merge shard-by-shard, avoiding a full base checkpoint in memory."""
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("merging HF checkpoints requires safetensors") from exc

    base_dir, output_dir = Path(base_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    updates, metadata = load_all_adapter_updates(adapter_dir)
    index_path = base_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        index = None
        shard_names = ["model.safetensors"]

    seen: set[str] = set()
    for shard_name in shard_names:
        shard = load_file(str(base_dir / shard_name), device="cpu")
        relevant = {key: updates[key] for key in shard if key in updates}
        merged = merge_updates_into_state(shard, relevant)
        seen.update(relevant)
        temporary = output_dir / f".{shard_name}.tmp"
        save_file(merged, str(temporary))
        os.replace(temporary, output_dir / shard_name)
        del shard, merged
    missing = sorted(set(updates) - seen)
    if missing:
        raise KeyError(f"base checkpoint is missing LoRA targets: {missing}")
    if index is not None:
        (output_dir / index_path.name).write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
    for path in base_dir.iterdir():
        if (
            path.is_file()
            and path.name not in set(shard_names)
            and path.name != "model.safetensors.index.json"
        ):
            shutil.copy2(path, output_dir / path.name)
    manifest = {
        "format": "geak_merged_hf_v1",
        "base_model": metadata["base_model"],
        "base_model_revision": metadata.get("base_model_revision"),
        "merged_weights": len(seen),
        "source_adapter": str(adapter_dir),
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
