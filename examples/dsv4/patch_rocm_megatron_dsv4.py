#!/usr/bin/env python3
"""Add DSV4 support to ROCm/Megatron-LM (rocm_dev branch).

Adds the minimal TransformerConfig fields and TransformerBlock/Layer hooks
needed by Lumen's ``get_dsv4_spec()`` to construct the DSV4 model.  The actual
DSV4 attention/compressor/indexer/HC layers come from the Lumen library — this
patch only makes Megatron's config and block infrastructure accept them.

Changes:
  1. TransformerConfig: add 'dsv4' to experimental_attention_variant Literal,
     add dsv4_* fields (hc_mult, compress_ratios, window_size, etc.),
     add dsv4_mode post_init flag.
  2. TransformerBlock: add HCHeadParams on last PP rank when dsv4_mode.
  3. TransformerLayer: add per-layer HC params (hc_attn_fn/base/scale,
     hc_ffn_fn/base/scale) when dsv4_mode.

Usage (inside container or on host):
    python3 examples/dsv4/patch_rocm_megatron_dsv4.py /path/to/Megatron-LM
"""

from __future__ import annotations

import os
import re
import sys


def patch_file(path: str, replacements: list[tuple[str, str]]) -> bool:
    with open(path) as f:
        content = f.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


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


def patch_moe_sqrtsoftplus(megatron_root: str) -> bool:
    """Add sqrtsoftplus MoE router score function (DeepSeek V4 default)."""
    changed = False

    cfg_path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_config.py")
    with open(cfg_path) as f:
        cfg = f.read()
    cfg_orig = cfg

    cfg = cfg.replace(
        "moe_router_score_function: Literal['softmax', 'sigmoid'] = \"softmax\"",
        "moe_router_score_function: Literal['softmax', 'sigmoid', 'sqrtsoftplus'] = \"softmax\"",
    )
    cfg = cfg.replace(
        '"""Score function for MoE routing. Can be "softmax" or "sigmoid"."""',
        '"""Score function for MoE routing. Can be "softmax", "sigmoid", or "sqrtsoftplus"."""',
    )
    cfg = cfg.replace(
        """        if self.moe_router_enable_expert_bias and self.moe_router_score_function != "sigmoid":
            raise ValueError(
                "Expert bias for aux-loss-free routing only supports sigmoid score function."
                "Please set --moe-router-score-function sigmoid for sigmoid score function."
            )""",
        """        if self.moe_router_enable_expert_bias and self.moe_router_score_function not in (
            "sigmoid",
            "sqrtsoftplus",
        ):
            raise ValueError(
                "Expert bias for aux-loss-free routing only supports sigmoid or sqrtsoftplus score function. "
                "Please set --moe-router-score-function to sigmoid or sqrtsoftplus."
            )""",
    )
    if cfg != cfg_orig:
        with open(cfg_path, "w") as f:
            f.write(cfg)
        changed = True

    moe_path = os.path.join(megatron_root, "megatron", "core", "transformer", "moe", "moe_utils.py")
    with open(moe_path) as f:
        moe = f.read()
    moe_orig = moe

    sqrtsoftplus_branch = """    elif score_function == "sqrtsoftplus":
        assert num_groups is None
        assert group_topk is None
        scores = torch.nn.functional.softplus(logits.float()).sqrt().type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
"""
    if 'elif score_function == "sqrtsoftplus"' not in moe:
        moe = moe.replace(
            """        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")""",
            """        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
""" + sqrtsoftplus_branch + """    else:
        raise ValueError(f"Invalid score_function: {score_function}")""",
        )
    if moe != moe_orig:
        with open(moe_path, "w") as f:
            f.write(moe)
        changed = True

    return changed


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


def patch_moe_router_freeze(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "moe", "router.py")
    with open(path) as f:
        content = f.read()
    orig = content

    if "moe_router_freeze_gate" not in content:
        content = content.replace(
            "        self.reset_parameters()\n",
            """        self.reset_parameters()

        if self.config.moe_router_freeze_gate:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
""",
            1,
        )
        content = content.replace(
            """        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        # Convert to specified datatype""",
            """        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        if self.config.moe_router_freeze_gate:
            assert not self.weight.requires_grad
            if self.bias is not None:
                assert not self.bias.requires_grad

        # Convert to specified datatype""",
            1,
        )

    if content != orig:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_dist_ckpt_skip_optional_dsv4_norms(megatron_root: str) -> bool:
    """Skip missing q_norm/kv_norm shards when flash ckpt omits late-layer norms.

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
                _optional_norm_suffixes = (
                    ".self_attention.q_norm.weight",
                    ".self_attention.kv_norm.weight",
                    ".self_attention.q_norm._norm.weight",
                    ".self_attention.kv_norm._norm.weight",
                )
                if any(sh_ten.key.endswith(suffix) for suffix in _optional_norm_suffixes):
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


def patch_shared_expert_clamp(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "moe", "shared_experts.py")
    with open(path) as f:
        content = f.read()
    orig = content

    needle = "        assert config.add_bias_linear == False, \"bias is not supported in the shared experts, \""
    if "activation_func_clamp_shared_expert" not in content and needle in content:
        content = content.replace(
            needle,
            """        if not config.activation_func_clamp_shared_expert:
            config.activation_func_clamp_value = None

"""
            + needle,
            1,
        )

    if content != orig:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_transformer_block(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_block.py")

    with open(path) as f:
        content = f.read()
    original = content

    # Add HC head params creation after _build_layers() call
    build_layers_call = "        self._build_layers()"
    if build_layers_call in content and "HCHeadParams" not in content:
        hc_block = '''
        # DSV4 Hyper-Connection head params (last PP rank only)
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.models.dsv4.ops.hyper_connection import HCHeadParams
            from megatron.core import parallel_state as mpu
            if mpu.is_pipeline_last_stage():
                self.hc_head_params = HCHeadParams(self.config)
'''
        content = content.replace(
            build_layers_call,
            build_layers_call + hc_block,
        )

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_transformer_layer(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_layer.py")

    with open(path) as f:
        content = f.read()
    original = content

    # Add per-layer HC params after self.mlp assignment in __init__
    # Find a stable anchor point — after the mlp is built
    if "dsv4_mode" not in content and "self.mlp = build_module" in content:
        # Find the class __init__ and add HC params
        # Look for the end of __init__ where self.mlp is assigned
        anchor = "        self.bias_dropout_add_exec_handler = torch.enable_grad"
        if anchor in content:
            hc_layer = '''
        # DSV4 Hyper-Connection per-layer params
        if getattr(self.config, 'dsv4_mode', False):
            import torch.nn as nn
            hc_mult = self.config.dsv4_hc_mult or 4
            hc_dim = hc_mult * self.config.hidden_size
            mix_size = (2 + hc_mult) * hc_mult
            self.hc_attn_fn = nn.Parameter(torch.zeros(mix_size, hc_dim, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.zeros(mix_size, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_size, hc_dim, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.zeros(mix_size, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            for p in [self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale,
                       self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale]:
                p._keep_fp32 = True

'''
            content = content.replace(anchor, hc_layer + anchor)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_eav_specs(megatron_root: str) -> bool:
    """Add dsv4 branch to get_experimental_attention_variant_module_spec."""
    path = os.path.join(megatron_root, "megatron", "core", "models", "gpt",
                        "experimental_attention_variant_module_specs.py")
    with open(path) as f:
        content = f.read()
    original = content

    # Replace the else branch to handle dsv4 before raising.
    # Lumen's get_dsv4_spec monkey-patches this at runtime, but the Literal
    # type needs to accept 'dsv4' without erroring.
    old_else = '''    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )'''
    new_else = '''    elif config.experimental_attention_variant == "dsv4":
        # DSV4 spec is injected by Lumen's get_dsv4_spec() monkey-patch at runtime
        raise ValueError(
            "DSV4 attention variant requires Lumen's get_dsv4_spec() — "
            "call it before get_experimental_attention_variant_module_spec()"
        )
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )'''
    if old_else in content and "dsv4" not in content:
        content = content.replace(old_else, new_else)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_tp_layers(megatron_root: str) -> bool:
    """Add condition_init_method to tensor_parallel/layers.py (needed by Lumen linears)."""
    path = os.path.join(megatron_root, "megatron", "core", "tensor_parallel", "layers.py")
    with open(path) as f:
        content = f.read()
    if "def condition_init_method" in content:
        return False
    stub = '''

def condition_init_method(config, init_method):
    """Condition weight initialization on config (Lumen compatibility shim).

    Returns the init_method unchanged — Lumen's LumenColumnParallelLinear calls
    this during CPU initialization. Xavier-uniform override is not used for DSV4.
    """
    if getattr(config, "init_method_xavier_uniform", False):
        import torch.nn.init as init
        return init.xavier_uniform_
    return init_method

'''
    content += stub
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_disable_batch_p2p_comm(megatron_root: str) -> bool:
    """Allow MEGATRON_NO_BATCH_P2P_COMM=1 to force batch_p2p_comm=False."""
    path = os.path.join(megatron_root, "megatron", "training", "arguments.py")
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        content = f.read()
    marker = "MEGATRON_NO_BATCH_P2P_COMM"
    if marker in content:
        return False
    old = "    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm\n"
    new = (
        "    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm\n"
        "    if os.environ.get('MEGATRON_NO_BATCH_P2P_COMM', '0') == '1':\n"
        "        kw_args['batch_p2p_comm'] = False\n"
    )
    if old not in content:
        return False
    with open(path, "w") as f:
        f.write(content.replace(old, new, 1))
    return True


def main(megatron_root: str) -> None:
    results = {
        "transformer_config.py": patch_transformer_config(megatron_root),
        "moe_sqrtsoftplus": patch_moe_sqrtsoftplus(megatron_root),
        "dsv4_training_config": patch_dsv4_training_config(megatron_root),
        "moe_router_freeze": patch_moe_router_freeze(megatron_root),
        "dist_ckpt_skip_dsv4_norms": patch_dist_ckpt_skip_optional_dsv4_norms(megatron_root),
        "shared_expert_clamp": patch_shared_expert_clamp(megatron_root),
        "transformer_block.py": patch_transformer_block(megatron_root),
        "transformer_layer.py": patch_transformer_layer(megatron_root),
        "experimental_attention_variant_module_specs.py": patch_eav_specs(megatron_root),
        "tensor_parallel/layers.py": patch_tp_layers(megatron_root),
        "disable_batch_p2p_comm": patch_disable_batch_p2p_comm(megatron_root),
    }
    print(f"Patched ROCm Megatron at {megatron_root}:")
    for name, ok in results.items():
        print(f"  {'PATCHED' if ok else 'skipped'}: {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <megatron-root>")
        sys.exit(1)
    main(sys.argv[1])
