from __future__ import annotations

import os
import re

def patch_transformer_block(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_block.py")

    with open(path) as f:
        content = f.read()
    original = content

    # Add HC utility and head params after _build_layers() call.
    build_layers_call = "        self._build_layers()"
    old_hc_block = '''
        # DSV4 Hyper-Connection head params (last PP rank only)
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.models.dsv4.hyper_connection import HCHeadParams
            from megatron.core import parallel_state as mpu
            if mpu.is_pipeline_last_stage():
                self.hc_head_params = HCHeadParams(self.config)
'''
    hc_block = '''
        # DSV4 Hyper-Connection state and learned output contraction.
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.models.dsv4.hyper_connection import (
                DeepSeekV4HyperConnectionUtil,
                HCHeadParams,
            )
            self.hc_util = DeepSeekV4HyperConnectionUtil(self.config)
            if self.post_process:
                self.hc_head_params = HCHeadParams(self.config)
'''
    legacy_ops_hc_block = '''
        # DSV4 Hyper-Connection head params (last PP rank only)
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.models.dsv4.ops.hyper_connection import HCHeadParams
            from megatron.core import parallel_state as mpu
            if mpu.is_pipeline_last_stage():
                self.hc_head_params = HCHeadParams(self.config)
'''
    legacy_models_hc_block = '''
        # DSV4 Hyper-Connection head params (last PP rank only)
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.models.dsv4.modules.hyper_connection import HCHeadParams
            from megatron.core import parallel_state as mpu
            if mpu.is_pipeline_last_stage():
                self.hc_head_params = HCHeadParams(self.config)
'''
    legacy_dsv4_hc_block = '''
        # DSV4 Hyper-Connection head params (last PP rank only)
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.dsv4.modules.hyper_connection import HCHeadParams
            from megatron.core import parallel_state as mpu
            if mpu.is_pipeline_last_stage():
                self.hc_head_params = HCHeadParams(self.config)
'''
    if legacy_ops_hc_block in content:
        content = content.replace(legacy_ops_hc_block, hc_block, 1)
    elif legacy_models_hc_block in content:
        content = content.replace(legacy_models_hc_block, hc_block, 1)
    elif legacy_dsv4_hc_block in content:
        content = content.replace(legacy_dsv4_hc_block, hc_block, 1)
    elif old_hc_block in content:
        content = content.replace(old_hc_block, hc_block, 1)
    elif build_layers_call in content and "self.hc_util =" not in content:
        content = content.replace(
            build_layers_call,
            build_layers_call + hc_block,
            1,
        )

    expand_anchor = (
        "        hidden_states = make_viewless_tensor("
        "inp=hidden_states, requires_grad=True, keep_graph=True)"
    )
    expand_block = '''

        # DSV4 mHC keeps four residual streams through every transformer layer.
        if getattr(self.config, 'dsv4_mode', False) and self.pre_process:
            hidden_states = self.hc_util.block_expand(hidden_states)
'''
    if (
        expand_anchor in content
        and "self.hc_util.block_expand(hidden_states)" not in content
    ):
        content = content.replace(
            expand_anchor, expand_anchor + expand_block, 1
        )

    head_anchor = "        # Final layer norm."
    head_block = '''        # Collapse DSV4 mHC streams before the final norm and LM head.
        if (
            getattr(self.config, 'dsv4_mode', False)
            and self.post_process
            and hasattr(self, "hc_head_params")
        ):
            hidden_states = self.hc_util.block_head(
                hidden_states,
                self.hc_head_params.hc_head_fn,
                self.hc_head_params.hc_head_scale,
                self.hc_head_params.hc_head_base,
            )

'''
    if head_anchor in content and "self.hc_util.block_head(" not in content:
        content = content.replace(head_anchor, head_block + head_anchor, 1)

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

    # Lumen's DSV4 attention returns a tensor, while Megatron BDA expects
    # the standard (output, bias) pair.
    attention_anchor = '        nvtx_range_pop(suffix="self_attention")'
    attention_compat = '''

        if isinstance(attention_output_with_bias, torch.Tensor):
            attention_output_with_bias = (attention_output_with_bias, None)
'''
    if (
        attention_anchor in content
        and "isinstance(attention_output_with_bias, torch.Tensor)" not in content
    ):
        content = content.replace(
            attention_anchor, attention_anchor + attention_compat, 1
        )

    attention_residual = '''        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm'''
    attention_pre = '''        # Residual connection.
        residual = hidden_states

        # DSV4 mHC contracts the residual streams for the attention sublayer.
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.models.dsv4.hyper_connection import (
                DeepSeekV4HyperConnectionUtil,
            )
            hc_util = DeepSeekV4HyperConnectionUtil(self.config)
            hidden_states, hc_attn_post, hc_attn_comb = hc_util.layer_pre(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
            )

        # Optional Input Layer norm'''
    if (
        attention_residual in content
        and "hc_attn_post, hc_attn_comb = hc_util.layer_pre" not in content
    ):
        content = content.replace(attention_residual, attention_pre, 1)

    attention_bda = '''        nvtx_range_push(suffix="self_attn_bda")
        if using_fused_tp_inference_kernel:'''
    attention_hc_post = '''        nvtx_range_push(suffix="self_attn_bda")
        if getattr(self.config, 'dsv4_mode', False):
            hidden_states = hc_util.layer_post(
                attention_output_with_bias,
                residual,
                hc_attn_post,
                hc_attn_comb,
            )
        elif using_fused_tp_inference_kernel:'''
    if (
        attention_bda in content
        and "                hc_attn_post,\n                hc_attn_comb," not in content
    ):
        content = content.replace(attention_bda, attention_hc_post, 1)

    mlp_pre_anchor = '''        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.'''
    mlp_pre = '''        # Residual connection.
        residual = hidden_states

        # DSV4 mHC contracts the residual streams for the MLP sublayer.
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.models.dsv4.hyper_connection import (
                DeepSeekV4HyperConnectionUtil,
            )
            hc_util = DeepSeekV4HyperConnectionUtil(self.config)
            hidden_states, hc_ffn_post, hc_ffn_comb = hc_util.layer_pre(
                hidden_states,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
            )

        # Optional Layer norm post the cross-attention.'''
    if (
        mlp_pre_anchor in content
        and "hc_ffn_post, hc_ffn_comb = hc_util.layer_pre" not in content
    ):
        content = content.replace(mlp_pre_anchor, mlp_pre, 1)

    post_call = "return self._forward_post_mlp(mlp_output_with_bias, residual)"
    post_call_hc = '''return self._forward_post_mlp(
                mlp_output_with_bias,
                residual,
                hc_ffn_post=hc_ffn_post if getattr(self.config, 'dsv4_mode', False) else None,
                hc_ffn_comb=hc_ffn_comb if getattr(self.config, 'dsv4_mode', False) else None,
            )'''
    if post_call in content and "hc_ffn_post=hc_ffn_post" not in content:
        content = content.replace(post_call, post_call_hc, 1)

    post_signature = (
        "    def _forward_post_mlp(self, mlp_output_with_bias, residual):"
    )
    post_signature_hc = '''    def _forward_post_mlp(
        self,
        mlp_output_with_bias,
        residual,
        *,
        hc_ffn_post=None,
        hc_ffn_comb=None,
    ):'''
    if post_signature in content:
        content = content.replace(post_signature, post_signature_hc, 1)

    mlp_bda = '''        nvtx_range_push(suffix="mlp_bda")
        if using_fused_tp_inference_kernel:'''
    mlp_hc_post = '''        nvtx_range_push(suffix="mlp_bda")
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.models.dsv4.hyper_connection import (
                DeepSeekV4HyperConnectionUtil,
            )
            hc_util = DeepSeekV4HyperConnectionUtil(self.config)
            hidden_states = hc_util.layer_post(
                mlp_output_with_bias,
                residual,
                hc_ffn_post,
                hc_ffn_comb,
            )
        elif using_fused_tp_inference_kernel:'''
    old_mlp_hc_post = '''        nvtx_range_push(suffix="mlp_bda")
        if getattr(self.config, 'dsv4_mode', False):
            hidden_states = hc_util.layer_post(
                mlp_output_with_bias,
                residual,
                hc_ffn_post,
                hc_ffn_comb,
            )
        elif using_fused_tp_inference_kernel:'''
    if old_mlp_hc_post in content:
        content = content.replace(old_mlp_hc_post, mlp_hc_post, 1)
    if (
        mlp_bda in content
        and "                hc_ffn_post,\n                hc_ffn_comb," not in content
    ):
        content = content.replace(mlp_bda, mlp_hc_post, 1)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False
