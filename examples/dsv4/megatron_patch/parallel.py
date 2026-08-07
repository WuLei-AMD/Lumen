from __future__ import annotations

import os
import re

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
