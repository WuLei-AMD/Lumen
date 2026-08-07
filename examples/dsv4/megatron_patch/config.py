from __future__ import annotations

import os
import re

def patch_transformer_config(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_config.py")

    with open(path) as f:
        content = f.read()
    original = content

    # 1. Add 'dsv4' to the Literal type
    content = content.replace(
        "Literal['gated_delta_net', 'dsa']",
        "Literal['gated_delta_net', 'dsa', 'dsv4']",
    )

    # 2. Add DSV4 fields after the DSA section
    dsa_marker = "    ####################\n    # DSA\n    ####################"
    if dsa_marker in content and "dsv4_mode: bool" not in content:
        last_dsa_match = None
        for m in re.finditer(r'    dsa_\w+:.*\n(?:    """.*?"""\n)?', content):
            last_dsa_match = m
        if last_dsa_match:
            insert_pos = last_dsa_match.end()
            dsv4_fields = '''
    ####################
    # DSV4
    ####################
    dsv4_mode: bool = False
    dsv4_hc_mult: Optional[int] = None
    dsv4_hc_sinkhorn_iters: int = 20
    dsv4_hc_eps: float = 1e-6
    dsv4_compress_ratios: Optional[List[int]] = None
    dsv4_compress_rope_theta: float = 160000.0
    dsv4_o_groups: Optional[int] = None
    dsv4_o_lora_rank: Optional[int] = None
    dsv4_n_hash_layers: int = 0
    dsv4_window_size: int = 128

'''
            content = content[:insert_pos] + dsv4_fields + content[insert_pos:]

    # 3. Add dsv4_mode = True in __post_init__ when variant == "dsv4"
    dsa_post_init = '        if self.experimental_attention_variant == "dsa":'
    if dsa_post_init in content and 'experimental_attention_variant == "dsv4"' not in content:
        dsv4_post_init = '''        if self.experimental_attention_variant == "dsv4":
            self.dsv4_mode = True

'''
        # Insert before the dsa check
        content = content.replace(dsa_post_init, dsv4_post_init + dsa_post_init)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False
def patch_dsv4_training_config(megatron_root: str) -> bool:
    """DSV4 finetune flags + typed compress ratios (ROCm Megatron gaps vs miles fork)."""
    cfg_path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_config.py")
    with open(cfg_path) as f:
        cfg = f.read()
    cfg_orig = cfg

    cfg = cfg.replace(
        "dsv4_compress_ratios: Optional[list] = None",
        "dsv4_compress_ratios: Optional[List[int]] = None",
    )
    if "activation_func_clamp_shared_expert:" not in cfg:
        cfg = cfg.replace(
            "    activation_func_clamp_value: Optional[float] = None",
            """    activation_func_clamp_value: Optional[float] = None
    activation_func_clamp_shared_expert: bool = True""",
            1,
        )
    if "freeze_e_score_correction_bias:" not in cfg:
        anchor = "    moe_router_enable_expert_bias: bool = False"
        cfg = cfg.replace(
            anchor,
            anchor
            + """
    freeze_e_score_correction_bias: bool = False
    moe_router_freeze_gate: bool = False""",
            1,
        )

    if cfg != cfg_orig:
        with open(cfg_path, "w") as f:
            f.write(cfg)
        return True
    return False
