"""Merge GEAK Sonic LoRA adapter into HF checkpoint for Qwen3-Coder ModuleList experts."""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def merge_adapter(base_dir: str, adapter_dir: str, output_dir: str):
    base_dir, adapter_dir, output_dir = Path(base_dir), Path(adapter_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(adapter_dir / "adapter_config.json") as f:
        cfg = json.load(f)

    ep_size = cfg.get("expert_parallel_size", cfg.get("world_size", 8))
    experts_per_rank = cfg.get("experts_per_rank", 16)
    modules = cfg.get("modules", {})

    adapter_shards = sorted(adapter_dir.glob("adapter_model.rank*.safetensors"))
    print(f"Loading {len(adapter_shards)} adapter shards...")

    all_deltas = {}
    for rank, path in enumerate(adapter_shards):
        state = load_file(str(path), device="cpu")
        ep_rank = rank % ep_size

        for module_name, meta in modules.items():
            a_key = f"{module_name}.lora_A"
            b_key = f"{module_name}.lora_B"
            if a_key not in state:
                continue
            lora_a = state[a_key].float()
            lora_b = state[b_key].float()
            alpha = float(meta["alpha"])
            r = int(meta["rank"])
            scale = alpha / r

            if "gate_up_lora" in module_name:
                prefix = module_name.replace(".local_experts.gate_up_lora", "")
                delta = torch.bmm(lora_b, lora_a) * scale
                gate_delta, up_delta = delta.chunk(2, dim=1)
                for local_i in range(delta.shape[0]):
                    global_i = ep_rank * experts_per_rank + local_i
                    gk = f"{prefix}.experts.{global_i}.gate_proj.weight"
                    uk = f"{prefix}.experts.{global_i}.up_proj.weight"
                    all_deltas[gk] = gate_delta[local_i]
                    all_deltas[uk] = up_delta[local_i]
            elif "down_lora" in module_name:
                prefix = module_name.replace(".local_experts.down_lora", "")
                delta = torch.bmm(lora_b, lora_a) * scale
                for local_i in range(delta.shape[0]):
                    global_i = ep_rank * experts_per_rank + local_i
                    dk = f"{prefix}.experts.{global_i}.down_proj.weight"
                    all_deltas[dk] = delta[local_i]
            else:
                delta = (lora_b @ lora_a) * scale
                all_deltas[f"{module_name}.weight"] = delta

        del state

    print(f"Computed {len(all_deltas)} weight deltas")

    index_path = base_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        index = None
        shard_names = ["model.safetensors"]

    merged_count = 0
    for shard_name in shard_names:
        print(f"Processing {shard_name}...")
        shard = load_file(str(base_dir / shard_name), device="cpu")
        for key in shard:
            if key in all_deltas:
                w = shard[key].float()
                d = all_deltas[key]
                if w.shape != d.shape:
                    raise ValueError(f"Shape mismatch for {key}: {w.shape} vs {d.shape}")
                shard[key] = (w + d).to(shard[key].dtype)
                merged_count += 1
        tmp = output_dir / f".{shard_name}.tmp"
        save_file(shard, str(tmp))
        os.replace(tmp, output_dir / shard_name)
        del shard

    print(f"Merged {merged_count}/{len(all_deltas)} deltas into base model")

    if index is not None:
        (output_dir / index_path.name).write_text(
            json.dumps(index, indent=2) + "\n"
        )

    for path in base_dir.iterdir():
        if path.is_file() and path.name not in set(shard_names) and path.name != "model.safetensors.index.json":
            if not (output_dir / path.name).exists():
                shutil.copy2(path, output_dir / path.name)

    manifest = {
        "base_model": cfg["base_model"],
        "base_model_revision": cfg.get("base_model_revision"),
        "merged_weights": merged_count,
        "training": "GEAK SFT 12K seq, 125 steps, FP8 blockwise2d, Sonic MoE EP=8",
    }
    (output_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Done!")
    return manifest


if __name__ == "__main__":
    merge_adapter(
        base_dir="/home/danyzhan/Lumen/experiments/GEAK-agent-coder/models/Qwen3-Coder-30B-A3B-Instruct",
        adapter_dir="/home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/qwen3-coder-12k-production/best",
        output_dir="/home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/qwen3-coder-12k-merged",
    )
