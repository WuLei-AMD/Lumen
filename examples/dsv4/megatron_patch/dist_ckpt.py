from __future__ import annotations

import os
import re

def patch_dist_ckpt_skip_optional_dsv4_norms(megatron_root: str) -> bool:
    """Skip missing optional shards when converted ckpt omits them.

    - q_norm/kv_norm on late layers (flash ckpt)
    - mlp.router.expert_bias (4-layer torch_dist from Miles convert)

    Enable with LUMEN_DSV4_SKIP_OPTIONAL_NORMS=1 (default). Set to 0 to fail on missing keys.
    """
    if os.environ.get("LUMEN_DSV4_SKIP_OPTIONAL_NORMS", "1") != "1":
        return False
    path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "dist_checkpointing",
        "strategies",
        "torch.py",
    )
    with open(path) as f:
        content = f.read()
    orig = content
    needle = """            if sh_ten.key not in metadata.state_dict_metadata:
                raise KeyError(
                    f"{sh_ten.key} from model not in state dict:"
                    f" {sorted(metadata.state_dict_metadata.keys())}"
                )"""
    if needle not in content:
        return False
    replacement = """            if sh_ten.key not in metadata.state_dict_metadata:
                _optional_skip_suffixes = (
                    ".self_attention.q_norm.weight",
                    ".self_attention.kv_norm.weight",
                    ".self_attention.q_norm._norm.weight",
                    ".self_attention.kv_norm._norm.weight",
                    ".mlp.router.expert_bias",
                )
                if any(sh_ten.key.endswith(suffix) for suffix in _optional_skip_suffixes):
                    try:
                        from megatron.training.utils import print_rank_0

                        print_rank_0(
                            f"{sh_ten.key} from model not in state dict, will skip"
                        )
                    except ImportError:
                        pass
                    continue
                raise KeyError(
                    f"{sh_ten.key} from model not in state dict:"
                    f" {sorted(metadata.state_dict_metadata.keys())}"
                )"""
    content = content.replace(needle, replacement, 1)
    if content != orig:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False
