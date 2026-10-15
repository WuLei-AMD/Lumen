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
    "atom_ops": ("lumen_atom_ops", "atom_ops"),
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
    # Comma-separated tags applied on top of rollout=ATOM. LUMEN_ATOM_OPS is merged in.
    # The checklist of what rollout=ATOM already replaces, and what each tag adds,
    # is qwen3-8b-lumen-toward-atom.html, section 二.
    # rmsnorm: Qwen3RMSNorm -> aiter.rmsnorm2d_fwd
    # residual_rmsnorm: residual add + RMSNorm -> aiter.rmsnorm2d_fwd_with_add
    # qkv: q_proj, k_proj, v_proj -> one tgemm.mm on the concatenated weight
    # rope: Qwen3 RoPE -> aiter rope_cached_positions_2c_fwd_inplace
    # varlen_attn: PyTorch SDPA -> aiter.flash_attn_varlen_func
    # gate_up: gate_proj, up_proj -> one tgemm.mm on the concatenated weight
    atom_ops: str = ""

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
            tags = self.atom_op_tags()
            if "residual_rmsnorm" in tags:
                tags.add("rmsnorm")
            self._patch_norms(
                model,
                atom_mode=True,
                extra_classes=("Qwen3RMSNorm",) if "rmsnorm" in tags else (),
            )
            try:
                self._patch_sdpa()
            except ImportError as exc:
                _rank0_print(f"> ATOM rollout: SDPA patch skipped ({exc})")
            if "varlen_attn" in tags:
                self._patch_varlen_attn()
            self._patch_linear(model, atom_mode=True)
            self._patch_mlp_activation(model)
            if "qkv" in tags:
                self._patch_qwen3_qkv(model)
            if "gate_up" in tags:
                self._patch_qwen3_gate_up(model)
            if "rope" in tags:
                self._patch_qwen3_rope(model)
            if "residual_rmsnorm" in tags:
                self._patch_qwen3_residual_rmsnorm(model)
            model._lumen_config = self
            tag_text = ",".join(sorted(tags)) if tags else "none"
            _rank0_print(f"> ATOM rollout mode: forward ops aligned with ATOM inference; atom_ops={tag_text}")
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

    def atom_op_tags(self) -> set[str]:
        """Tags that turn on one ATOM operator replacement at a time."""
        import os

        known = {"rmsnorm", "residual_rmsnorm", "qkv", "rope", "varlen_attn", "gate_up"}
        raw = ",".join(part for part in (self.atom_ops, os.environ.get("LUMEN_ATOM_OPS", "")) if part)
        tags = {part.strip() for part in raw.split(",") if part.strip()}
        unknown = tags - known
        if unknown:
            _rank0_print(f"> ATOM atom_ops ignored unknown tags: {', '.join(sorted(unknown))}")
        return tags & known

    def _patch_norms(self, model, *, atom_mode: bool = False, extra_classes: tuple[str, ...] = ()) -> None:
        import torch

        if atom_mode:
            import os

            if os.environ.get("ATOM_USE_TORCH_RMSNORM", "0") == "1":
                _rank0_print("> ATOM rollout: ATOM_USE_TORCH_RMSNORM=1, keeping original RMSNorm")
                return

            from aiter import rmsnorm2d_fwd, rmsnorm2d_fwd_with_add

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

            class _AtomAddRMSNormFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, residual, weight, eps):
                    x_2d = x.reshape(-1, x.shape[-1])
                    residual_2d = residual.reshape(-1, residual.shape[-1])
                    out = torch.empty_like(x_2d)
                    residual_out = torch.empty_like(x_2d)
                    rmsnorm2d_fwd_with_add(
                        out,
                        x_2d,
                        residual_2d,
                        residual_out,
                        weight,
                        float(eps),
                        use_model_sensitive_rmsnorm=0,
                    )
                    ctx.save_for_backward(residual_out, weight)
                    ctx.eps = float(eps)
                    ctx.shape = x.shape
                    return out.view(x.shape), residual_out.view(x.shape)

                @staticmethod
                def backward(ctx, grad_out, grad_residual):
                    residual_out, weight = ctx.saved_tensors
                    from aiter.ops.triton.normalization.rmsnorm import _rmsnorm_backward

                    grad_2d = grad_out.reshape(-1, grad_out.shape[-1])
                    variance = residual_out.float().pow(2).mean(dim=-1)
                    rsigma = torch.rsqrt(variance + ctx.eps)
                    dx, dw = _rmsnorm_backward(grad_2d, residual_out, weight, rsigma)
                    if grad_residual is not None:
                        dx = dx + grad_residual.reshape(-1, grad_residual.shape[-1])
                    dx = dx.view(ctx.shape)
                    return dx, dx, dw, None

            class _AtomRMSNorm(torch.nn.Module):
                def __init__(self, weight, eps):
                    super().__init__()
                    self.weight = torch.nn.Parameter(weight.data.clone())
                    self.eps = eps

                def forward(self, x, residual=None):
                    if residual is None:
                        return _AtomRMSNormFn.apply(x, self.weight, self.eps)
                    return _AtomAddRMSNormFn.apply(x, residual, self.weight, self.eps)

            _ATOM_RMSNORM_CLASSES = (
                "RMSNorm",
                "LlamaRMSNorm",
                "MistralRMSNorm",
                "Qwen2RMSNorm",
                "MegatronRMSNorm",
                "TENorm",
                "LumenRMSNorm",
                "_MegatronCompatibleTLNorm",
                "MegatronCompatibleTLNorm",
                "_MegatronCompatibleTLRMSNorm",
                "MegatronCompatibleTLRMSNorm",
                *extra_classes,
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

    def _patch_qwen3_qkv(self, model) -> None:
        """Replace Qwen3 q/k/v projections with one concatenated GEMM.

        ``q_proj`` runs ``tgemm.mm`` on ``cat(q, k, v)`` weights. ``k_proj`` and
        ``v_proj`` return the slices from that same result, so Q/K RMSNorm still
        see the tensors ATOM splits out of ``qkv_proj``.
        """
        import torch
        from aiter.tuned_gemm import TunedGemm

        tgemm = TunedGemm()

        class _AtomQKVFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, wq, wk, wv, bq, bk, bv):
                weight = torch.cat((wq, wk, wv), dim=0)
                if bq is None and bk is None and bv is None:
                    bias = None
                else:
                    parts = [
                        part_bias if part_bias is not None else weight.new_zeros(part.shape[0])
                        for part, part_bias in ((wq, bq), (wk, bk), (wv, bv))
                    ]
                    bias = torch.cat(parts, dim=0)
                ctx.save_for_backward(x, wq, wk, wv)
                ctx.bias_flags = (bq is not None, bk is not None, bv is not None)
                return tgemm.mm(x, weight, bias, otype=x.dtype)

            @staticmethod
            def backward(ctx, grad_output):
                x, wq, wk, wv = ctx.saved_tensors
                weight = torch.cat((wq, wk, wv), dim=0)
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
                input_flat = x.reshape(-1, x.shape[-1])
                grad_input = (grad_flat @ weight).reshape(x.shape)
                grad_weight = grad_flat.t() @ input_flat
                sizes = (wq.shape[0], wk.shape[0], wv.shape[0])
                gq, gk, gv = grad_weight.split(sizes, dim=0)
                if any(ctx.bias_flags):
                    gbq, gbk, gbv = grad_flat.sum(0).split(sizes, dim=0)
                    gbq = gbq if ctx.bias_flags[0] else None
                    gbk = gbk if ctx.bias_flags[1] else None
                    gbv = gbv if ctx.bias_flags[2] else None
                else:
                    gbq = gbk = gbv = None
                return grad_input, gq, gk, gv, gbq, gbk, gbv

        count = 0
        for module in model.modules():
            if type(module).__name__ != "Qwen3Attention":
                continue
            q_mod = getattr(module, "q_proj", None)
            k_mod = getattr(module, "k_proj", None)
            v_mod = getattr(module, "v_proj", None)
            if q_mod is None or k_mod is None or v_mod is None:
                continue
            pending = {}

            def q_forward(hidden, q_mod=q_mod, k_mod=k_mod, v_mod=v_mod, pending=pending):
                fused = _AtomQKVFn.apply(
                    hidden, q_mod.weight, k_mod.weight, v_mod.weight, q_mod.bias, k_mod.bias, v_mod.bias
                )
                q, k, v = fused.split(
                    (q_mod.weight.shape[0], k_mod.weight.shape[0], v_mod.weight.shape[0]),
                    dim=-1,
                )
                pending["k"] = k
                pending["v"] = v
                return q

            def k_forward(hidden, k_mod=k_mod, pending=pending):
                cached = pending.pop("k", None)
                if cached is not None:
                    return cached
                return tgemm.mm(hidden, k_mod.weight, k_mod.bias, otype=hidden.dtype)

            def v_forward(hidden, v_mod=v_mod, pending=pending):
                cached = pending.pop("v", None)
                if cached is not None:
                    return cached
                return tgemm.mm(hidden, v_mod.weight, v_mod.bias, otype=hidden.dtype)

            q_mod.forward = q_forward
            k_mod.forward = k_forward
            v_mod.forward = v_forward
            count += 1
        if count:
            _rank0_print(f"> qkv: {count} Qwen3 attention blocks use one tgemm.mm")

    def _patch_qwen3_gate_up(self, model) -> None:
        """Replace Qwen3 gate/up projections with one concatenated GEMM.

        ``gate_proj`` runs ``tgemm.mm`` on ``cat(gate, up)`` weights. ``up_proj``
        returns the second half of that result, so the MLP still sees the two
        slices ATOM splits out of ``gate_up_proj``.
        """
        import torch
        from aiter.tuned_gemm import TunedGemm

        tgemm = TunedGemm()

        class _AtomGateUpFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w_gate, w_up, b_gate, b_up):
                weight = torch.cat((w_gate, w_up), dim=0)
                if b_gate is None and b_up is None:
                    bias = None
                else:
                    parts = [
                        part_bias if part_bias is not None else weight.new_zeros(part.shape[0])
                        for part, part_bias in ((w_gate, b_gate), (w_up, b_up))
                    ]
                    bias = torch.cat(parts, dim=0)
                ctx.save_for_backward(x, w_gate, w_up)
                ctx.bias_flags = (b_gate is not None, b_up is not None)
                return tgemm.mm(x, weight, bias, otype=x.dtype)

            @staticmethod
            def backward(ctx, grad_output):
                x, w_gate, w_up = ctx.saved_tensors
                weight = torch.cat((w_gate, w_up), dim=0)
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
                input_flat = x.reshape(-1, x.shape[-1])
                grad_input = (grad_flat @ weight).reshape(x.shape)
                grad_weight = grad_flat.t() @ input_flat
                sizes = (w_gate.shape[0], w_up.shape[0])
                g_gate, g_up = grad_weight.split(sizes, dim=0)
                if any(ctx.bias_flags):
                    gb_gate, gb_up = grad_flat.sum(0).split(sizes, dim=0)
                    gb_gate = gb_gate if ctx.bias_flags[0] else None
                    gb_up = gb_up if ctx.bias_flags[1] else None
                else:
                    gb_gate = gb_up = None
                return grad_input, g_gate, g_up, gb_gate, gb_up

        count = 0
        for module in model.modules():
            if type(module).__name__ != "Qwen3MLP":
                continue
            gate_mod = getattr(module, "gate_proj", None)
            up_mod = getattr(module, "up_proj", None)
            if gate_mod is None or up_mod is None:
                continue
            pending = {}

            def gate_forward(hidden, gate_mod=gate_mod, up_mod=up_mod, pending=pending):
                fused = _AtomGateUpFn.apply(
                    hidden, gate_mod.weight, up_mod.weight, gate_mod.bias, up_mod.bias
                )
                gate, up = fused.split((gate_mod.weight.shape[0], up_mod.weight.shape[0]), dim=-1)
                pending["up"] = up
                return gate

            def up_forward(hidden, up_mod=up_mod, pending=pending):
                cached = pending.pop("up", None)
                if cached is not None:
                    return cached
                return tgemm.mm(hidden, up_mod.weight, up_mod.bias, otype=hidden.dtype)

            gate_mod.forward = gate_forward
            up_mod.forward = up_forward
            count += 1
        if count:
            _rank0_print(f"> gate_up: {count} Qwen3 MLP blocks use one tgemm.mm")

    def _patch_qwen3_rope(self, model) -> None:
        """Apply ATOM's cached RoPE on Qwen3 Q and K, then keep HF's head layout.

        Q and K stay ``(batch, tokens, heads, dim)`` for the kernel. The result
        is transposed back to ``(batch, heads, tokens, dim)`` for the rest of
        the Hugging Face attention forward.
        """
        import torch
        from aiter.rotary_embedding import _apply_rotary_emb, get_rope

        def _make_rope_fn(rotary):
            class _AtomRoPEFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, q, k, positions):
                    nonlocal rotary
                    if rotary.cos_cache.device != q.device:
                        # ATOM constructs its RoPE cache under the GPU default
                        # device.  Recomputing there matters: a CPU-computed
                        # cache first differs at position 64 for Qwen3-8B.
                        with torch.device(q.device):
                            rotary = type(rotary)(
                                rotary.head_size,
                                rotary.rotary_dim,
                                rotary.max_position_embeddings,
                                rotary.base,
                                rotary.is_neox_style,
                                q.dtype,
                            )
                    elif rotary.cos_cache.dtype != q.dtype:
                        rotary.cos_cache = rotary.cos_cache.to(device=q.device, dtype=q.dtype)
                        rotary.sin_cache = rotary.sin_cache.to(device=q.device, dtype=q.dtype)
                    ctx.rotary = rotary
                    ctx.save_for_backward(positions)
                    q_out = q.detach().clone()
                    k_out = k.detach().clone()
                    return rotary(positions, q_out, k_out)

                @staticmethod
                def backward(ctx, grad_q, grad_k):
                    positions, = ctx.saved_tensors
                    index = positions.reshape(-1)
                    cos = ctx.rotary.cos_cache.index_select(0, index).squeeze(-1).squeeze(-1)
                    sin = ctx.rotary.sin_cache.index_select(0, index).squeeze(-1).squeeze(-1)
                    grad_q = _apply_rotary_emb(grad_q, cos, -sin, True)
                    grad_k = _apply_rotary_emb(grad_k, cos, -sin, True)
                    return grad_q, grad_k, None

            return _AtomRoPEFn

        count = 0
        for module in model.modules():
            if type(module).__name__ != "Qwen3Attention":
                continue
            cfg = module.config
            rope_type = cfg.rope_parameters.get("rope_type", "default")
            if rope_type != "default":
                _rank0_print(f"> rope: skip {rope_type}")
                continue
            head_dim = int(module.head_dim)
            rotary = get_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=int(cfg.max_position_embeddings),
                base=cfg.rope_parameters["rope_theta"],
                is_neox_style=True,
                dtype=torch.bfloat16,
            )
            rope_fn = _make_rope_fn(rotary)

            def forward(
                hidden_states,
                position_embeddings=None,
                attention_mask=None,
                past_key_values=None,
                position_ids=None,
                module=module,
                rope_fn=rope_fn,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)
                query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape))
                key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape))
                value = module.v_proj(hidden_states).view(hidden_shape)
                if position_ids is None:
                    position_ids = torch.arange(
                        query.shape[1], device=query.device
                    ).expand(query.shape[0], -1)
                positions = position_ids.reshape(-1).to(dtype=torch.long)
                query_tok = query.reshape(-1, query.shape[-2], query.shape[-1])
                key_tok = key.reshape(-1, key.shape[-2], key.shape[-1])
                query_tok, key_tok = rope_fn.apply(query_tok, key_tok, positions)
                kwargs = dict(kwargs)
                kwargs.pop("position_ids", None)
                query_states = query_tok.view_as(query).transpose(1, 2)
                key_states = key_tok.view_as(key).transpose(1, 2)
                value_states = value.transpose(1, 2)
                from transformers.models.qwen3.modeling_qwen3 import (
                    ALL_ATTENTION_FUNCTIONS,
                    eager_attention_forward,
                )

                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, module.layer_idx
                    )
                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    module.config._attn_implementation, eager_attention_forward
                )
                attn_output, attn_weights = attention_interface(
                    module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not module.training else module.attention_dropout,
                    scaling=module.scaling,
                    sliding_window=module.sliding_window,
                    **kwargs,
                )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = module.o_proj(attn_output)
                return attn_output, attn_weights

            module.forward = forward
            count += 1
        if count:
            _rank0_print(f"> rope: {count} Qwen3 attention blocks use rope_cached_positions_2c_fwd_inplace")

    def _patch_qwen3_residual_rmsnorm(self, model) -> None:
        """Carry ATOM's residual into the next RMSNorm instead of adding in PyTorch.

        Layer 0 input norm stays a plain ``rmsnorm2d_fwd``. Every later norm that
        ATOM fuses with the residual — post-attention, the next layer input, and
        the final norm — calls ``rmsnorm2d_fwd_with_add``.
        """
        backbone = getattr(model, "model", model)
        layers = getattr(backbone, "layers", None)
        norm = getattr(backbone, "norm", None)
        if not layers or norm is None or type(layers[0]).__name__ != "Qwen3DecoderLayer":
            _rank0_print("> residual_rmsnorm: Qwen3 decoder stack not found, skipped")
            return

        backbone._atom_residual = None

        def layer_forward(layer):
            def forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                position_embeddings=None,
                **kwargs,
            ):
                carried = backbone._atom_residual
                backbone._atom_residual = None
                if carried is None:
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = layer.input_layernorm(hidden_states, carried)
                hidden_states, _ = layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
                hidden_states = layer.mlp(hidden_states)
                backbone._atom_residual = residual
                return hidden_states

            return forward

        for layer in layers:
            layer.forward = layer_forward(layer)

        orig_norm = norm.forward
        orig_model_forward = backbone.forward

        def norm_forward(hidden_states, residual=None):
            carried = backbone._atom_residual if residual is None else residual
            backbone._atom_residual = None
            if carried is None:
                return orig_norm(hidden_states)
            out, _ = orig_norm(hidden_states, carried)
            return out

        def model_forward(*args, **kwargs):
            backbone._atom_residual = None
            return orig_model_forward(*args, **kwargs)

        norm.forward = norm_forward
        backbone.forward = model_forward
        _rank0_print(
            f"> residual_rmsnorm: {len(layers)} Qwen3 layers use rmsnorm2d_fwd_with_add"
        )

    def _patch_sdpa(self) -> None:
        """Monkey-patch ``F.scaled_dot_product_attention`` with AITER attention."""
        from lumen.ops.attention.hf_patch import patch_sdpa

        patch_sdpa()

    def _patch_varlen_attn(self) -> None:
        """Replace causal SDPA with ATOM's prefill kernel, ``flash_attn_varlen_func``.

        Hugging Face calls SDPA with ``(batch, heads, tokens, dim)``. The varlen
        kernel takes ``(tokens, heads, dim)`` plus cumulative sequence lengths.
        """
        import torch
        import torch.nn.functional as F

        import aiter

        original = F.scaled_dot_product_attention

        def varlen_sdpa(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            **kwargs,
        ):
            if attn_mask is not None or dropout_p != 0.0 or not is_causal or query.dim() != 4:
                return original(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    **kwargs,
                )
            batch, heads, tokens, dim = query.shape
            kv_tokens = key.shape[2]
            q = query.permute(0, 2, 1, 3).reshape(batch * tokens, heads, dim).contiguous()
            k = key.permute(0, 2, 1, 3).reshape(batch * kv_tokens, key.shape[1], key.shape[-1]).contiguous()
            v = value.permute(0, 2, 1, 3).reshape(batch * kv_tokens, value.shape[1], value.shape[-1]).contiguous()
            cu_q = torch.arange(0, (batch + 1) * tokens, tokens, device=query.device, dtype=torch.int32)
            cu_k = torch.arange(0, (batch + 1) * kv_tokens, kv_tokens, device=query.device, dtype=torch.int32)
            softmax_scale = dim ** -0.5 if scale is None else float(scale)
            out = aiter.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=tokens,
                max_seqlen_k=kv_tokens,
                # Match ATOM's prefill metadata.  This value selects the
                # skip-zero FMHA variant and is not merely a shape hint.
                min_seqlen_q=0,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=(-1, -1, 0),
            )
            return out.view(batch, tokens, heads, dim).permute(0, 2, 1, 3).contiguous()

        F.scaled_dot_product_attention = varlen_sdpa
        _rank0_print("> varlen_attn: causal SDPA uses aiter.flash_attn_varlen_func")

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
            for _name, module in model.named_modules():
                if isinstance(module, nn.Linear):

                    def _make_atom_forward(mod):
                        def _forward(input):
                            return _AtomLinearFn.apply(input, mod.weight, mod.bias)
                        return _forward

                    module.forward = _make_atom_forward(module)
                    count += 1

            if count:
                _rank0_print(f"> Replaced {count} nn.Linear forward with ATOM TunedGemm.mm()")

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

    def _patch_mlp_activation(self, model) -> None:
        """Replace HF MLP forward with ATOM's fused ``aiter.silu_and_mul``."""
        import torch

        from aiter import silu_and_mul

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
                        gate = m.gate_proj(x)
                        up = m.up_proj(x)
                        gate_up = torch.cat([gate, up], dim=-1)
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
