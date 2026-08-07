from __future__ import annotations

import os
import re

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
