###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unified Lumen configuration — single dataclass covering quantization,
attention, norm replacement, and execution/fusion features.

Usage::

    from lumen.config import LumenConfig

    # Programmatic — flat kwargs, one call
    cfg = LumenConfig(format="fp8_e4m3", scaling="delayed",
                      fp8_attn="dpa", lumen_norm=True, fused_mlp=True)
    cfg.enable(model)

    # From CLI argparse namespace
    cfg = LumenConfig.from_args(args)
    cfg.enable(model, dp_group=dp_group)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, replace
from typing import Optional

logger = logging.getLogger(__name__)


o_proj_linear_count = 0


def _o_proj_linear(activation, weight, bias):
    """o_proj GEMM. The tuned ASM split-K kernel is not bitwise across launches."""
    global o_proj_linear_count
    import torch.nn.functional as F

    o_proj_linear_count += 1
    if activation.dim() >= 3:
        flat = activation.reshape(-1, activation.size(-1))
        out = F.linear(flat, weight, bias)
        return out.view(*activation.shape[:-1], weight.shape[0])
    return F.linear(activation, weight, bias)


def _rank0_print(msg: str) -> None:
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    logger.info(msg)


# ---------------------------------------------------------------------------
# Arg-name mapping: LumenConfig field → argparse attribute name(s)
# ---------------------------------------------------------------------------

_ARG_MAP: dict[str, tuple[str, ...]] = {
    "format": ("linear_fp8_format", "lumen_fp8_format"),
    "scaling": ("linear_fp8_scaling",),
    "block_size": ("linear_fp8_block_size",),
    "amax_algo": ("linear_fp8_amax_algo",),
    "margin": ("linear_fp8_margin",),
    "reduce_amax": ("linear_fp8_reduce_amax",),
    "history_len": ("linear_fp8_amax_history",),
    "quantize_activation": ("linear_fp8_activation",),
    "fp8_wgrad": ("linear_fp8_wgrad",),
    "quantize_grad": ("grad_quant_type",),
    "first_last_layers_bf16": ("first_last_layers_bf16",),
    "num_layers_at_start_in_bf16": ("num_layers_at_start_in_bf16",),
    "num_layers_at_end_in_bf16": ("num_layers_at_end_in_bf16",),
    "num_layers": ("num_layers",),
    "use_sdma": ("use_sdma",),
    "fp8_attn": ("lumen_fp8_attn",),
    "attn_backend": ("lumen_attn_backend",),
    "attn_quant_type": ("lumen_fp8_quant_type",),
    "lumen_norm": ("lumen_norm",),
    "hf_attn_patch": ("hf_attn_patch",),
    "lumen_linear": ("lumen_linear",),
    "fused_mlp": ("lumen_fused_mlp",),
    "fp8_activation_store": ("lumen_fp8_activation_store",),
    "cpu_offload": ("lumen_cpu_offload",),
    "delay_wgrad": ("lumen_delay_wgrad",),
    "gradient_accumulation_fusion": ("lumen_gradient_accumulation_fusion",),
    "fp8_param_gather": ("lumen_fp8_param_gather",),
    "fp8_weight_cache": ("lumen_fp8_weight_cache",),
    "fused_rope": ("lumen_fused_rope",),
    "hip_graphs": ("lumen_hip_graphs",),
    "fp8_checkpoint": ("lumen_fp8_checkpoint",),
    "fp8_param_manager": ("fp8_param_manager",),
    "lora_rank": ("lora_rank",),
    "lora_alpha": ("lora_alpha",),
    "lora_dropout": ("lora_dropout",),
    "use_8bit_adam": ("use_8bit_adam",),
    "rollout": ("lumen_rollout",),
}


@dataclass
class LumenConfig:
    """Unified configuration for all Lumen training features.

    Groups functionality behind a single flat interface:

    * **Tier 0 — Weight storage & adapters:** ``fp8_param_manager``,
      ``lora_rank`` / ``lora_alpha`` / ``lora_dropout``.  Applied first
      (FP8ParamManager before LoRA) so adapter weights stay BF16.
    * **Tier 1 — Linear FP8:** ``format``, ``scaling``, ``block_size``, etc.
      These are forwarded to :class:`~lumen.quantize.QuantConfig`.
    * **Tier 2 — Attention FP8 & norms:** ``fp8_attn``, ``attn_backend``,
      ``attn_quant_type``, ``lumen_norm``.
    * **Tier 3 — Execution / fusion:** ``fused_mlp``, ``cpu_offload``,
      ``delay_wgrad``, ``hip_graphs``, etc.
    * **Optimizer hint:** ``use_8bit_adam`` — not applied to the model,
      read by the trainer to select 8-bit Adam from bitsandbytes.
    """

    # -- Tier 1: Linear FP8 (forwarded to QuantConfig) --
    format: str = "fp8_e4m3"
    scaling: str = "delayed"
    block_size: int = 128
    amax_algo: str = "max"
    margin: int = 0
    reduce_amax: bool = False
    history_len: int = 16
    quantize_activation: bool = True
    fp8_wgrad: bool = True
    quantize_grad: Optional[str] = None
    first_last_layers_bf16: bool = False
    num_layers_at_start_in_bf16: int = 1
    num_layers_at_end_in_bf16: int = 1
    num_layers: int = 0
    use_sdma: bool = False

    # -- Tier 2: Attention FP8 --
    fp8_attn: str = "none"
    attn_backend: str = "auto"
    attn_quant_type: str = "blockwise"

    # -- Tier 2: Norm replacement --
    lumen_norm: bool = False

    # -- Tier 2: Attention patching --
    hf_attn_patch: bool = False

    # -- Tier 2: Linear GEMM patching (BF16) --
    lumen_linear: bool = False

    # -- Tier 3: Execution / fusion --
    fused_mlp: bool = False
    fp8_activation_store: bool = False
    cpu_offload: bool = False
    delay_wgrad: bool = False
    gradient_accumulation_fusion: bool = False
    fp8_param_gather: bool = False
    fp8_weight_cache: bool = False
    fused_rope: bool = False
    hip_graphs: bool = False
    fp8_checkpoint: bool = False

    # -- Rollout: forward-op alignment with inference engines --
    rollout: str = ""  # "" = default Lumen ops, "ATOM" = align forward with ATOM inference

    # -- Tier 0: FP8 weight storage (applied before everything else) --
    fp8_param_manager: bool = False

    # -- Tier 0: LoRA via HuggingFace PEFT (after FP8ParamManager) --
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # -- Optimizer hint (not applied to model, read by trainer) --
    use_8bit_adam: bool = False

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def quant_config(self):
        """Build the inner :class:`~lumen.quantize.QuantConfig`."""
        from lumen.quantize import QuantConfig

        return QuantConfig.from_str(
            format=self.format,
            scaling=self.scaling,
            block_size=self.block_size,
            amax_algo=self.amax_algo,
            margin=self.margin,
            reduce_amax=self.reduce_amax,
            history_len=self.history_len,
            quantize_activation=self.quantize_activation,
            fp8_wgrad=self.fp8_wgrad,
            quantize_grad=self.quantize_grad,
            first_last_layers_bf16=self.first_last_layers_bf16,
            num_layers_at_start_in_bf16=self.num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=self.num_layers_at_end_in_bf16,
            num_layers=self.num_layers,
            use_sdma=self.use_sdma,
            fp8_dpa=self.fp8_attn in ("dpa", "mha"),
            fp8_mha=self.fp8_attn == "mha",
        )

    # -----------------------------------------------------------------------
    # Orchestrated enablement
    # -----------------------------------------------------------------------

    @property
    def has_any_features(self) -> bool:
        """Return True if any Lumen feature is enabled."""
        return (
            bool(self.rollout)
            or self.quant_config.is_quantized
            or self.lumen_norm
            or self.hf_attn_patch
            or self.lumen_linear
            or self.fp8_param_manager
            or self.lora_rank > 0
            or self.fused_mlp
            or self.fp8_activation_store
            or self.cpu_offload
            or self.fp8_param_gather
            or self.fp8_weight_cache
            or self.fp8_checkpoint
            or self.fused_rope
            or self.hip_graphs
            or self.delay_wgrad
            or self.gradient_accumulation_fusion
        )

    def enable(self, model, *, dp_group=None, backend: str = "auto"):
        """Apply all Lumen features to *model* in the correct order.

        Orchestration:
          0a. FP8ParamManager — quantize linear weights to FP8 storage
          0b. LoRA (PEFT) — wrap linears with trainable adapters
          1.  Norm patching (before quant so new norm modules get patched)
          2.  Pre-quant module flags (delay_wgrad, grad-accum fusion, etc.)
          3.  ``quant.enable()`` — FP8 linear patching
          4.  Post-quant features (fp8_checkpoint, fp8_param_gather)
          5.  Attach config to model for downstream reads

        FP8ParamManager runs **before** LoRA so that only base ``nn.Linear``
        weights are quantized; the LoRA adapter weights (``lora_A``, ``lora_B``)
        created afterwards stay in BF16 and remain trainable.

        Returns:
            ``(manager, model)`` — the :class:`~lumen.quantize.ScalingManager`
            (or ``None``) and the model (may be a new PEFT wrapper).
        """
        import torch

        if self.rollout.upper() == "ATOM":
            try:
                from lumenrl.core.atom_train_align import bind_lumen_rollout

                bind_lumen_rollout()
            except ImportError:
                pass
            self._patch_norms(model, atom_mode=True)
            self._patch_residual_stream(model)
            self._patch_rope(model)
            self._patch_sdpa()
            self._patch_atom_attention()
            self._patch_linear(model, atom_mode=True)
            self._patch_merged_qkv(model)
            self._patch_mlp_activation(model)
            model._lumen_config = self
            _rank0_print("> ATOM rollout mode: forward ops aligned with ATOM inference")
            return None, model

        # 0a. FP8 param storage (replaces weight.data with FP8, freezes)
        fp8pm_mgr = None
        if self.fp8_param_manager:
            fp8pm_mgr = self._apply_fp8_param_manager(model)

        # 0b. LoRA adapters (PEFT)
        if self.lora_rank > 0:
            model = self._apply_lora(model)

        # 1. Norm patching
        if self.lumen_norm:
            self._patch_norms(model)

        # 1b. Attention patching (monkey-patch F.scaled_dot_product_attention)
        if self.hf_attn_patch:
            self._patch_sdpa()

        # 1c. Linear GEMM patching (BF16 AITER Triton)
        if self.lumen_linear:
            self._patch_linear(model)

        # 2. Pre-quant module attributes
        self._apply_pre_quant(model)

        # 3. FP8 linear quantization
        qcfg = self.quant_config
        manager = None
        if qcfg.is_quantized:
            import torch.distributed as dist

            import lumen.quantize as quant

            if dp_group is None and qcfg.reduce_amax and dist.is_initialized():
                dp_group = dist.group.WORLD
            manager = quant.enable(
                model,
                config=qcfg,
                backend=backend,
                dp_group=dp_group if qcfg.reduce_amax else None,
            )

        # 4. Post-quant features
        self._apply_post_quant(model, manager)

        # 5. Attach config + FP8ParamManager reference
        model._lumen_config = self
        if fp8pm_mgr is not None:
            model._fp8_param_manager = fp8pm_mgr

        self._log_summary(manager)
        return manager, model

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _apply_fp8_param_manager(self, model):
        """Replace linear weights with FP8 storage via FP8ParamManager.

        Targets ``nn.Linear`` and Megatron ``ColumnParallelLinear`` /
        ``RowParallelLinear``.  Must run before LoRA wrapping so that
        adapter weights stay BF16/trainable.
        """
        import torch

        from lumen.quantize.fp8_params import FP8ParamManager

        mgr = FP8ParamManager(fp8_dtype=torch.float8_e4m3fn)
        count = mgr.quantize_params(model)
        hooks = mgr.register_dequant_hooks(model)
        _rank0_print(f"> FP8ParamManager: quantized {count} params, registered {hooks} hooks")
        return mgr

    def _apply_lora(self, model):
        """Apply LoRA adapters via HuggingFace PEFT and freeze the base model.

        When FP8ParamManager is active, PEFT's ``_move_adapter_to_device_of_base_layer``
        casts LoRA adapter weights to FP8 (matching the quantized base weight dtype).
        We fix this by re-casting all LoRA adapter parameters back to BF16 so they
        remain trainable with standard arithmetic.
        """
        import torch
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(self.lora_target_modules),
        )
        model = get_peft_model(model, peft_config)

        if self.fp8_param_manager:
            fixed = 0
            for name, param in model.named_parameters():
                if "lora_" in name and param.dtype in (
                    torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                    torch.float8_e5m2, torch.float8_e5m2fnuz,
                ):
                    param.data = param.data.to(torch.bfloat16)
                    param.requires_grad_(True)
                    fixed += 1
            if fixed:
                _rank0_print(f"> Fixed {fixed} LoRA params cast to FP8 by PEFT -> BF16")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        _rank0_print(
            f"> LoRA applied (rank={self.lora_rank}, alpha={self.lora_alpha}) — "
            f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )
        return model

    def _patch_residual_stream(self, model) -> None:
        """Thread the residual through RMSNorm the way ATOM's Qwen3 decoder does."""
        inner = getattr(model, "model", None)
        if inner is None or type(inner).__name__ != "Qwen3Model":
            return

        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
        from transformers.modeling_outputs import BaseModelOutputWithPast

        def _layer(layer, hidden, residual, attention_mask, position_embeddings, position_ids, past_key_values, use_cache, kwargs):
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            hidden, _ = layer.self_attn(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = layer.mlp(hidden)
            return hidden, residual

        def forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            **kwargs,
        ):
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = inner.embed_tokens(input_ids)
            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=inner.config)
            if position_ids is None:
                past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
                position_ids = position_ids.unsqueeze(0)
            if not isinstance(causal_mask_mapping := attention_mask, dict):
                mask_kwargs = {
                    "config": inner.config,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }
                causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
                if inner.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

            hidden = inputs_embeds
            position_embeddings = inner.rotary_emb(hidden, position_ids)
            residual = None
            for i, layer in enumerate(inner.layers[: inner.config.num_hidden_layers]):
                hidden, residual = _layer(
                    layer,
                    hidden,
                    residual,
                    causal_mask_mapping[inner.config.layer_types[i]],
                    position_embeddings,
                    position_ids,
                    past_key_values,
                    use_cache,
                    kwargs,
                )
            hidden, _residual = inner.norm(hidden, residual)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden,
                past_key_values=past_key_values if use_cache else None,
            )

        inner.forward = forward
        _rank0_print("> Qwen3 residual stream uses aiter.rmsnorm2d_fwd_with_add")

    def _patch_rope(self, model) -> None:
        """Qwen3 RoPE: same kernel and cos/sin cache as ATOM ``RotaryEmbedding``."""
        import contextvars

        import torch

        config = getattr(model, "config", None)
        if config is None or getattr(config, "model_type", None) != "qwen3":
            return

        from transformers.models.qwen3 import modeling_qwen3

        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        rope_params = getattr(config, "rope_parameters", None) or {}
        base = float(rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
        max_pos = int(config.max_position_embeddings)
        positions_ctx: contextvars.ContextVar = contextvars.ContextVar(
            "atom_rope_positions", default=None
        )
        cache: dict = {}
        orig_apply = modeling_qwen3.apply_rotary_pos_emb

        def _cos_sin(device, dtype):
            key = (str(device), dtype)
            if key not in cache:
                inv_freq = 1.0 / (
                    base ** (
                        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                        / head_dim
                    )
                )
                t = torch.arange(max_pos, dtype=torch.float32, device=device)
                freqs = torch.einsum("i,j->ij", t, inv_freq)
                cache[key] = (
                    freqs.cos().unsqueeze(-2).unsqueeze(-2).to(dtype),
                    freqs.sin().unsqueeze(-2).unsqueeze(-2).to(dtype),
                )
            return cache[key]

        def _atom_apply(q, k, cos, sin, unsqueeze_dim=1):
            pos = positions_ctx.get()
            if pos is None:
                return orig_apply(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
            from aiter import rope_cached_positions_2c_fwd_inplace

            b, h, s, d = q.shape
            q_tok = q.transpose(1, 2).reshape(b * s, h, d).contiguous()
            k_tok = k.transpose(1, 2).reshape(b * s, k.shape[1], d).contiguous()
            num_tokens = q_tok.shape[0]
            q_in = q_tok.view(1, num_tokens, h, d)
            k_in = k_tok.view(1, num_tokens, k_tok.shape[1], d)
            positions = pos.to(device=q.device).reshape(1, num_tokens)
            cos_cache, sin_cache = _cos_sin(q.device, q.dtype)
            rope_cached_positions_2c_fwd_inplace(
                q_in,
                k_in,
                cos_cache,
                sin_cache,
                positions,
                0,
                reuse_freqs_front_part=True,
                nope_first=False,
            )
            q_out = q_tok.view(b, s, h, d).transpose(1, 2).contiguous()
            k_out = k_tok.view(b, s, k_tok.shape[1], d).transpose(1, 2).contiguous()
            return q_out, k_out

        modeling_qwen3.apply_rotary_pos_emb = _atom_apply
        wrapped = 0
        for module in model.modules():
            if type(module).__name__ != "Qwen3Attention":
                continue
            orig_fwd = module.forward

            def _make(fwd):
                def _forward(*args, **kwargs):
                    token = positions_ctx.set(kwargs.get("position_ids"))
                    try:
                        return fwd(*args, **kwargs)
                    finally:
                        positions_ctx.reset(token)
                return _forward

            module.forward = _make(orig_fwd)
            wrapped += 1
        if wrapped:
            _rank0_print(
                f"> Replaced {wrapped} Qwen3 RoPE calls with "
                "aiter.rope_cached_positions_2c_fwd_inplace"
            )

    def _patch_norms(self, model, *, atom_mode: bool = False) -> None:
        import torch

        if atom_mode:
            import os

            if os.environ.get("ATOM_USE_TORCH_RMSNORM", "0") == "1":
                _rank0_print("> ATOM rollout: ATOM_USE_TORCH_RMSNORM=1, keeping original RMSNorm")
                return

            from aiter import rmsnorm2d_fwd

            class _AtomRMSNormFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, weight, eps):
                    ctx.save_for_backward(x, weight)
                    ctx.eps = eps
                    x_2d = x.reshape(-1, x.shape[-1])
                    y = rmsnorm2d_fwd(x_2d, weight, eps)
                    return y.reshape(x.shape)

                @staticmethod
                def backward(ctx, grad_output):
                    x, weight = ctx.saved_tensors
                    from aiter.ops.triton.normalization.rmsnorm import _rmsnorm_backward

                    x_2d = x.reshape(-1, x.shape[-1])
                    go_2d = grad_output.reshape(-1, grad_output.shape[-1])
                    dx, dw = _rmsnorm_backward(go_2d, x_2d, weight, ctx.eps)
                    return dx.reshape(x.shape), dw, None

            class _AtomRMSNorm(torch.nn.Module):
                def __init__(self, weight, eps):
                    super().__init__()
                    self.weight = torch.nn.Parameter(weight.data.clone())
                    self.eps = eps

                def forward(self, x, residual=None):
                    if residual is None:
                        return _AtomRMSNormFn.apply(x, self.weight, self.eps)
                    from aiter import rmsnorm2d_fwd_with_add

                    dim = self.weight.shape[0]
                    x_2d = x.reshape(-1, dim).contiguous()
                    residual_2d = residual.reshape(-1, dim).contiguous()
                    out = torch.empty_like(x_2d)
                    residual_out = torch.empty_like(x_2d)
                    rmsnorm2d_fwd_with_add(
                        out, x_2d, residual_2d, residual_out, self.weight, self.eps
                    )
                    return out.view(x.shape), residual_out.view(x.shape)

            _ATOM_RMSNORM_CLASSES = (
                "RMSNorm",
                "LlamaRMSNorm",
                "MistralRMSNorm",
                "Qwen2RMSNorm",
                "Qwen3RMSNorm",
                "MegatronRMSNorm",
                "TENorm",
                "LumenRMSNorm",
                "_MegatronCompatibleTLNorm",
                "MegatronCompatibleTLNorm",
                "_MegatronCompatibleTLRMSNorm",
                "MegatronCompatibleTLRMSNorm",
            )
            count = 0
            for _name, module in model.named_modules():
                for attr_name, child in list(module.named_children()):
                    cls_name = type(child).__name__
                    if cls_name in _ATOM_RMSNORM_CLASSES:
                        w = getattr(child, "weight", None)
                        if w is None:
                            w = getattr(getattr(child, "_norm", None), "weight", None)
                        if w is None:
                            continue
                        eps = getattr(child, "eps", getattr(child, "variance_epsilon",
                              getattr(child, "epsilon", 1e-6)))
                        repl = _AtomRMSNorm(w, eps)
                        if w.is_meta:
                            repl.to(device="meta")
                        setattr(module, attr_name, repl)
                        count += 1
            if count:
                _rank0_print(f"> Replaced {count} RMSNorm modules with ATOM-aligned aiter.rmsnorm2d_fwd")
            return

        from lumen.ops.normalization import LumenLayerNorm, LumenRMSNorm

        count = 0
        for _name, module in model.named_modules():
            for attr_name, child in list(module.named_children()):
                cls_name = type(child).__name__
                if cls_name in (
                    "RMSNorm",
                    "LlamaRMSNorm",
                    "MistralRMSNorm",
                    "Qwen2RMSNorm",
                ):
                    hidden = child.weight.shape[0]
                    eps = getattr(child, "eps", getattr(child, "variance_epsilon", 1e-6))
                    target_dtype = child.weight.dtype if not child.weight.is_meta else torch.bfloat16
                    repl = LumenRMSNorm(hidden, eps=eps, grad_quant_type=self.quantize_grad)
                    if not child.weight.is_meta:
                        repl.weight.data.copy_(child.weight.data)
                    repl.to(target_dtype)
                    if child.weight.is_meta:
                        repl.to(device="meta")
                    setattr(module, attr_name, repl)
                    count += 1
                elif cls_name in ("LayerNorm",):
                    hidden = child.weight.shape[0] if child.weight is not None else child.normalized_shape[0]
                    eps = getattr(child, "eps", 1e-5)
                    repl = LumenLayerNorm(hidden, eps=eps, grad_quant_type=self.quantize_grad)
                    is_meta = child.weight is not None and child.weight.is_meta
                    target_dtype = child.weight.dtype if (child.weight is not None and not is_meta) else torch.bfloat16
                    if child.weight is not None and not is_meta:
                        repl.weight.data.copy_(child.weight.data)
                    if hasattr(child, "bias") and child.bias is not None and repl.bias is not None and not is_meta:
                        repl.bias.data.copy_(child.bias.data)
                    repl.to(target_dtype)
                    if is_meta:
                        repl.to(device="meta")
                    setattr(module, attr_name, repl)
                    count += 1

        if count:
            _rank0_print(f"> Replaced {count} norm modules with Lumen implementations")

    def _patch_sdpa(self) -> None:
        """Monkey-patch ``F.scaled_dot_product_attention`` with AITER attention."""
        from lumen.ops.attention.hf_patch import patch_sdpa

        patch_sdpa()

    @staticmethod
    def _auto_tune_bf16_gemm(model) -> None:
        """Placeholder — BF16 GEMM uses torch.mm fallback (safe on MI350).

        hipBLASLt and ASM JIT both SIGABRT on MI350 gfx950, so we skip
        tuned GEMM injection and let ``tuned_gemm.gemm_a16w16`` fall back
        to ``torch.mm`` (which internally uses hipBLASLt via PyTorch).
        """
        pass

    def _patch_linear(self, model, *, atom_mode: bool = False) -> None:
        """Replace ``nn.Linear`` forward with AITER GEMM (BF16, autograd-safe).

        When ``atom_mode=True``, uses ``TunedGemm().mm()`` (same as ATOM inference).
        Otherwise uses ``dispatch_gemm`` (Lumen default dispatch).
        """
        import torch
        import torch.nn as nn

        if atom_mode:
            from aiter.tuned_gemm import TunedGemm

            tgemm = TunedGemm()

            class _AtomLinearFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input, weight, bias):
                    ctx.save_for_backward(input, weight)
                    ctx.has_bias = bias is not None
                    return tgemm.mm(input, weight, bias, otype=input.dtype)

                @staticmethod
                def backward(ctx, grad_output):
                    input, weight = ctx.saved_tensors
                    grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
                    input_flat = input.reshape(-1, input.shape[-1])
                    grad_input = grad_flat @ weight
                    grad_input = grad_input.reshape(input.shape)
                    grad_weight = grad_flat.t() @ input_flat
                    grad_bias = (grad_flat.sum(0) if ctx.has_bias else None)
                    return grad_input, grad_weight, grad_bias

            # HF path: patch nn.Linear modules
            count = 0
            o_count = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):

                    def _make_atom_forward(mod, linear_name):
                        def _forward(input):
                            if linear_name.endswith("o_proj"):
                                return _o_proj_linear(input, mod.weight, mod.bias)
                            return _AtomLinearFn.apply(input, mod.weight, mod.bias)
                        return _forward

                    module.forward = _make_atom_forward(module, name)
                    count += 1
                    if name.endswith("o_proj"):
                        o_count += 1

            if count:
                _rank0_print(
                    f"> Replaced {count} nn.Linear forward with ATOM TunedGemm.mm(); "
                    f"{o_count} o_proj use F.linear"
                )

            # Megatron path: patch _do_gemm only if model contains Lumen parallel linears
            try:
                from lumen.modules.parallel_linear import (
                    LumenColumnParallelLinear,
                    LumenRowParallelLinear,
                )
                has_parallel = any(
                    isinstance(m, (LumenColumnParallelLinear, LumenRowParallelLinear))
                    for m in model.modules()
                )
            except ImportError:
                has_parallel = False

            if has_parallel:
                import lumen.modules.parallel_linear as _pl

                _orig_do_gemm = _pl._do_gemm

                def _atom_do_gemm(
                    input_, weight, bias, scaling_manager, scaling_type,
                    fp8_dtype, block_size, gradient_accumulation_fusion=False,
                    delay_wgrad=False, deferred_wgrad=None,
                    activation_tensor_id=None, pre_quantized_input=None,
                ):
                    if scaling_type != "none" or delay_wgrad:
                        return _orig_do_gemm(
                            input_, weight, bias, scaling_manager, scaling_type,
                            fp8_dtype, block_size, gradient_accumulation_fusion,
                            delay_wgrad, deferred_wgrad, activation_tensor_id,
                            pre_quantized_input,
                        )
                    return tgemm.mm(input_, weight, bias, otype=input_.dtype)

                _pl._do_gemm = _atom_do_gemm
                _rank0_print("> ATOM rollout: patched parallel linear GEMM to use TunedGemm.mm()")

            return

        self._auto_tune_bf16_gemm(model)

        from lumen.ops.quantize.linear import dispatch_gemm

        class _GemmA16W16Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, weight, bias):
                ctx.save_for_backward(input, weight)
                ctx.has_bias = bias is not None
                output = dispatch_gemm(input, weight, scaling_type="none", bias=bias)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                input, weight = ctx.saved_tensors
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
                input_flat = input.reshape(-1, input.shape[-1])
                grad_input = grad_flat @ weight
                grad_input = grad_input.reshape(input.shape)
                grad_weight = grad_flat.t() @ input_flat
                grad_bias = (grad_flat.sum(0) if ctx.has_bias else None)
                return grad_input, grad_weight, grad_bias

        count = 0
        for _name, module in model.named_modules():
            if isinstance(module, nn.Linear):

                def _make_aiter_forward(mod):
                    def _aiter_forward(input):
                        return _GemmA16W16Fn.apply(input, mod.weight, mod.bias)

                    return _aiter_forward

                module.forward = _make_aiter_forward(module)
                count += 1

        if count:
            _rank0_print(f"> Replaced {count} nn.Linear forward with AITER GEMM (ASM→HIP→Triton)")

    def _patch_atom_attention(self) -> None:
        """Qwen3 SDPA path: ATOM prefill's ``flash_attn_varlen_func``."""
        import torch
        from aiter import flash_attn_varlen_func
        from transformers.modeling_utils import AttentionInterface

        def _varlen(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=None,
            is_causal=None,
            **kwargs,
        ):
            del attention_mask, kwargs
            b, hq, seqlen, dim = query.shape
            hk = key.shape[1]
            q = query.transpose(1, 2).reshape(b * seqlen, hq, dim).contiguous()
            k = key.transpose(1, 2).reshape(b * seqlen, hk, dim).contiguous()
            v = value.transpose(1, 2).reshape(b * seqlen, value.shape[1], dim).contiguous()
            cu = torch.arange(
                0, (b + 1) * seqlen, seqlen, device=query.device, dtype=torch.int32
            )
            sw = getattr(module, "sliding_window", None)
            window_size = (-1, 0, 0) if sw is None or sw == -1 else (sw, 0, 0)
            # ATOM prefill sets causal=True. HF folds a dense causal mask into
            # is_causal=False before SDPA; the module flag is the contract.
            del is_causal
            causal = seqlen > 1 and bool(getattr(module, "is_causal", True))
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=seqlen,
                max_seqlen_k=seqlen,
                min_seqlen_q=seqlen,
                dropout_p=dropout,
                softmax_scale=scaling if scaling is not None else dim ** -0.5,
                causal=causal,
                window_size=window_size,
                sink_ptr=None,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out.view(b, seqlen, hq, dim), None

        AttentionInterface.register("sdpa", _varlen)
        _rank0_print("> Qwen3 SDPA routed to aiter.flash_attn_varlen_func")

    def _patch_merged_qkv(self, model) -> None:
        """One QKV GEMM per layer, weight stacked as q then k then v."""
        import torch
        from aiter.tuned_gemm import tgemm

        count = 0
        for module in model.modules():
            if type(module).__name__ != "Qwen3Attention":
                continue
            q_proj, k_proj, v_proj = module.q_proj, module.k_proj, module.v_proj
            sizes = (q_proj.out_features, k_proj.out_features, v_proj.out_features)
            box: dict = {}

            def _bias(x, layers):
                if all(layer.bias is None for layer in layers):
                    return None
                parts = []
                for layer in layers:
                    if layer.bias is None:
                        parts.append(torch.zeros(layer.out_features, device=x.device, dtype=x.dtype))
                    else:
                        parts.append(layer.bias)
                return torch.cat(parts, dim=0)

            def _q(x, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, sizes=sizes, box=box):
                layers = (q_proj, k_proj, v_proj)
                weight = torch.cat([layer.weight for layer in layers], dim=0)
                y = tgemm.mm(x, weight, _bias(x, layers), otype=x.dtype)
                q_out, k_out, v_out = y.split(sizes, dim=-1)
                box["k"] = k_out
                box["v"] = v_out
                return q_out

            def _k(_x, box=box):
                return box.pop("k")

            def _v(_x, box=box):
                return box.pop("v")

            q_proj.forward = _q
            k_proj.forward = _k
            v_proj.forward = _v
            count += 1
        if count:
            _rank0_print(f"> Merged {count} Qwen3 QKV projections into one TunedGemm.mm")

    def _patch_mlp_activation(self, model) -> None:
        """Replace HF MLP forward with ATOM's fused ``aiter.silu_and_mul``."""
        import torch

        from aiter import silu_and_mul
        from aiter.tuned_gemm import tgemm

        class _AtomSiluAndMulFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                out = torch.empty(
                    [*x.shape[:-1], x.shape[-1] // 2],
                    device=x.device, dtype=x.dtype,
                )
                silu_and_mul(out, x)
                return out

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                d = x.shape[-1] // 2
                gate, up = x[..., :d], x[..., d:]
                sig = torch.sigmoid(gate)
                d_gate = grad_output * up * (sig + gate * sig * (1.0 - sig))
                d_up = grad_output * (gate * sig)
                return torch.cat([d_gate, d_up], dim=-1)

        # HF path: modules with gate_proj/up_proj/down_proj/act_fn (e.g. Qwen3MLP)
        hf_count = 0
        for _name, module in model.named_modules():
            if (hasattr(module, 'gate_proj') and hasattr(module, 'up_proj')
                    and hasattr(module, 'down_proj') and hasattr(module, 'act_fn')):

                def _make_atom_mlp_forward(m):
                    def _forward(x):
                        layers = (m.gate_proj, m.up_proj)
                        weight = torch.cat([layer.weight for layer in layers], dim=0)
                        if all(layer.bias is None for layer in layers):
                            bias = None
                        else:
                            parts = []
                            for layer in layers:
                                if layer.bias is None:
                                    parts.append(torch.zeros(
                                        layer.out_features, device=x.device, dtype=x.dtype
                                    ))
                                else:
                                    parts.append(layer.bias)
                            bias = torch.cat(parts, dim=0)
                        gate_up = tgemm.mm(x, weight, bias, otype=x.dtype)
                        hidden = _AtomSiluAndMulFn.apply(gate_up)
                        return m.down_proj(hidden)
                    return _forward

                module.forward = _make_atom_mlp_forward(module)
                hf_count += 1

        if hf_count:
            _rank0_print(f"> Replaced {hf_count} HF MLP forward with ATOM fused silu_and_mul")

        # Megatron path: MLP with linear_fc1/linear_fc2 + gated_linear_unit
        meg_count = 0
        try:
            from megatron.core.transformer.mlp import MLP as _MegatronMLP
        except ImportError:
            _MegatronMLP = None

        if _MegatronMLP is not None:
            for _name, module in model.named_modules():
                if not isinstance(module, _MegatronMLP):
                    continue
                if not getattr(module.config, "gated_linear_unit", False):
                    continue

                def _make_atom_meg_mlp_forward(m):
                    def _forward(hidden_states, per_token_scale=None):
                        intermediate, bias = m.linear_fc1(hidden_states)
                        if bias is not None:
                            intermediate = intermediate + bias
                        intermediate = _AtomSiluAndMulFn.apply(intermediate)
                        output, output_bias = m.linear_fc2(intermediate)
                        return output, output_bias
                    return _forward

                module.forward = _make_atom_meg_mlp_forward(module)
                meg_count += 1

        if meg_count:
            _rank0_print(f"> Replaced {meg_count} Megatron MLP forward with ATOM fused silu_and_mul")

    def _apply_pre_quant(self, model) -> None:
        """Set module attributes that must exist before ``quant.enable()``.

        ``delay_wgrad`` and ``gradient_accumulation_fusion`` are only
        meaningful on Lumen-native parallel linear types.

        ``fp8_activation_store`` is set on both Lumen-native types **and**
        standard ``nn.Linear`` so that ``_replace_forward`` captures the
        flag and passes it into ``QuantizedLinearFunction``.
        """
        import torch.nn as nn

        has_lumen_only = self.delay_wgrad or self.gradient_accumulation_fusion
        has_act_store = self.fp8_activation_store

        if not (has_lumen_only or has_act_store):
            return

        lumen_count = 0
        if has_lumen_only:
            try:
                from lumen.modules.grouped_linear import LumenGroupedLinear
                from lumen.modules.layernorm_linear import LumenLayerNormLinear
                from lumen.modules.parallel_linear import (
                    LumenColumnParallelLinear,
                    LumenRowParallelLinear,
                )

                lumen_types = (
                    LumenColumnParallelLinear,
                    LumenRowParallelLinear,
                    LumenLayerNormLinear,
                    LumenGroupedLinear,
                )
            except ImportError:
                lumen_types = ()

            for module in model.modules():
                if lumen_types and isinstance(module, lumen_types):
                    if self.delay_wgrad and hasattr(module, "delay_wgrad"):
                        module.delay_wgrad = True
                    if self.gradient_accumulation_fusion and hasattr(module, "gradient_accumulation_fusion"):
                        module.gradient_accumulation_fusion = True
                    if has_act_store:
                        module.fp8_activation_store = True
                    lumen_count += 1

        linear_count = 0
        if has_act_store:
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.fp8_activation_store = True
                    linear_count += 1

        opts = []
        if self.delay_wgrad:
            opts.append("delay_wgrad")
        if self.gradient_accumulation_fusion:
            opts.append("gradient_accumulation_fusion")
        if has_act_store:
            opts.append("fp8_activation_store")
        total = lumen_count + linear_count
        if total:
            _rank0_print(
                f"> Pre-quant optimizations ({', '.join(opts)}) applied to "
                f"{total} modules ({lumen_count} Lumen, {linear_count} nn.Linear)"
            )

    def _apply_post_quant(self, model, manager) -> None:
        """Apply features that require the ScalingManager from ``quant.enable()``."""
        if self.fp8_checkpoint:
            if manager is not None:
                self._enable_fp8_checkpoint(manager)
            else:
                _rank0_print("> WARNING: fp8_checkpoint requires FP8 quantization")

        if self.fp8_param_gather:
            if manager is not None:
                manager.enable_fp8_params(model)
                _rank0_print(f"> FP8 param gather enabled ({manager.num_fp8_params} params)")
            else:
                _rank0_print("> WARNING: fp8_param_gather requires FP8 quantization")

        if self.fp8_weight_cache:
            if manager is not None:
                import lumen.quantize as quant

                count = quant.store_weights_fp8(model)
                _rank0_print(f"> FP8 weight cache enabled ({count} layers cached)")
            else:
                _rank0_print("> WARNING: fp8_weight_cache requires FP8 quantization")

    @staticmethod
    def _enable_fp8_checkpoint(manager) -> None:
        """Monkey-patch Megatron checkpoint to preserve FP8 scaling state.

        No-op if Megatron is not installed.
        """
        try:
            import megatron.core.tensor_parallel as tp_module
            import megatron.core.tensor_parallel.random as tp_random
        except ImportError:
            return

        from lumen.utils.checkpoint import _FP8ScalingContext

        if hasattr(tp_module, "_lumen_fp8_checkpoint_patched"):
            return

        _original = tp_random.checkpoint

        def _patched(function, distribute_saved_activations, *args):
            ctx = _FP8ScalingContext()
            ctx.save(manager)
            orig_fn = function

            def wrapped(*a, **kw):
                ctx.restore(manager)
                return orig_fn(*a, **kw)

            return _original(wrapped, distribute_saved_activations, *args)

        tp_random.checkpoint = _patched
        tp_module.checkpoint = _patched
        tp_module._lumen_fp8_checkpoint_patched = True
        tp_module._lumen_fp8_checkpoint_original = _original
        _rank0_print("> FP8-aware activation checkpointing enabled")

    def _log_summary(self, manager) -> None:
        parts = [f"format={self.format}", f"scaling={self.scaling}"]
        if self.fp8_param_manager:
            parts.append("fp8_param_manager")
        if self.lora_rank > 0:
            parts.append(f"lora_rank={self.lora_rank}")
        if self.fp8_attn != "none":
            parts.append(f"fp8_attn={self.fp8_attn}")
        if self.lumen_norm:
            parts.append("lumen_norm")
        if self.hf_attn_patch:
            parts.append("hf_attn_patch")
        if self.lumen_linear:
            parts.append("lumen_linear")
        if self.use_8bit_adam:
            parts.append("use_8bit_adam")
        tier3 = [
            name
            for name in (
                "fused_mlp",
                "cpu_offload",
                "delay_wgrad",
                "gradient_accumulation_fusion",
                "fp8_param_gather",
                "fp8_weight_cache",
                "fused_rope",
                "hip_graphs",
                "fp8_checkpoint",
                "fp8_activation_store",
            )
            if getattr(self, name, False)
        ]
        if tier3:
            parts.append(f"features=[{', '.join(tier3)}]")
        _rank0_print(f"> LumenConfig enabled ({', '.join(parts)})")

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> LumenConfig:
        """Build from an argparse ``Namespace`` (replaces manual ``getattr`` blocks).

        Iterates :data:`_ARG_MAP` to find matching attributes on *args*.
        Unknown / missing attributes are silently skipped (defaults apply).

        Respects the ``linear_fp8`` boolean gate: when it is ``False``, the
        FP8 Linear quantization fields (format, scaling, etc.) are suppressed
        so that ``quant_config.is_quantized`` remains ``False``.
        """
        linear_fp8_enabled = getattr(args, "linear_fp8", None)

        kwargs: dict = {}
        for field_name, arg_names in _ARG_MAP.items():
            for arg_name in arg_names:
                val = getattr(args, arg_name, None)
                if val is not None:
                    kwargs[field_name] = val
                    break

        if linear_fp8_enabled is False:
            kwargs["scaling"] = "none"

        return cls(**kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> LumenConfig:
        """Build from a plain dict (e.g. YAML / JSON config)."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
