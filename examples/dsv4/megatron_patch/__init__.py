"""Modular ROCm Megatron DSV4 patch steps."""

from __future__ import annotations

PATCH_VERSION = "20260812"

from .config import patch_dsv4_training_config, patch_transformer_config
from .dist_ckpt import patch_dist_ckpt_skip_optional_dsv4_norms
from .eav import patch_eav_specs
from .hc import patch_transformer_block, patch_transformer_layer
from .moe import (
    patch_moe_router_freeze,
    patch_moe_sqrtsoftplus,
    patch_shared_expert_clamp,
)
from .parallel import patch_disable_batch_p2p_comm, patch_tp_layers

PATCH_STEPS = (
    ("transformer_config.py", patch_transformer_config),
    ("moe_sqrtsoftplus", patch_moe_sqrtsoftplus),
    ("dsv4_training_config", patch_dsv4_training_config),
    ("moe_router_freeze", patch_moe_router_freeze),
    ("dist_ckpt_skip_dsv4_norms", patch_dist_ckpt_skip_optional_dsv4_norms),
    ("shared_expert_clamp", patch_shared_expert_clamp),
    ("transformer_block.py", patch_transformer_block),
    ("transformer_layer.py", patch_transformer_layer),
    ("experimental_attention_variant_module_specs.py", patch_eav_specs),
    ("tensor_parallel/layers.py", patch_tp_layers),
    ("disable_batch_p2p_comm", patch_disable_batch_p2p_comm),
)


def apply_all(megatron_root: str) -> dict[str, bool]:
    return {name: fn(megatron_root) for name, fn in PATCH_STEPS}


def main(megatron_root: str) -> None:
    print(f"Patched ROCm Megatron at {megatron_root} (patch_version={PATCH_VERSION}):")
    for name, ok in apply_all(megatron_root).items():
        print(f"  {'PATCHED' if ok else 'skipped'}: {name}")
