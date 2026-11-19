"""Thin LoRA SFT loop composed from Lumen's Qwen3 MoE/FSDP primitives."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from geak_agent_coder.config import (
    assert_runtime_compatible,
    validate_launch_environment,
)

from .checkpoint import load_distributed_adapter, save_adapter
from .data import (
    LengthBucketDistributedSampler,
    LossMaskCollator,
    PretokenizedLossMaskDataset,
    masked_causal_lm_loss_sum,
)
from .launcher import assert_training_authorized, load_config
from .lora import LoRAConfig, audit_trainable_parameters, inject_hierarchical_lora

LOGGER = logging.getLogger(__name__)


def _lumen_args(config: dict) -> SimpleNamespace:
    """Map project config to the narrow argument contract used by Lumen."""
    model = config["model"]
    distributed = config["distributed"]
    precision = config["precision"]
    return SimpleNamespace(
        model_name_or_path=model["name_or_path"],
        max_position_embeddings=model.get("max_position_embeddings", 40960),
        norm_topk_prob=True,
        aux_loss_coeff=0.0,
        fuse_rope=config.get("optimizations", {}).get("fuse_rope", False),
        mode=precision["mode"],
        expert_backend=distributed["expert_backend"],
        fp8_scaling=precision.get("scaling", precision.get("base_scaling", "none")),
        fp8_block_size=int(precision.get("block_size", 128)),
        fp8_format=precision.get("fp8_format"),
        lumen_format=precision.get("lumen_format", "fp8_e4m3"),
        aiter_attn=config.get("optimizations", {}).get("aiter_attn", False),
        lumen_norm=config.get("optimizations", {}).get("lumen_norm", False),
        fused_router=config.get("optimizations", {}).get("fused_router", False),
        lumen_moe_dispatch_overlap=config.get("optimizations", {}).get(
            "moe_dispatch_overlap", False
        ),
        lumen_moe_global_expert_layout=config.get("optimizations", {}).get(
            "moe_global_expert_layout", False
        ),
        sharding=distributed.get("sharding", "full_shard"),
        fp8_param_storage=precision.get("fp8_param_storage", False),
    )


def _enable_lumen(model, args, dp_group):
    """Apply Lumen using the profile's precision parameters."""

    if (
        args.mode == "bf16"
        and not args.aiter_attn
        and not args.lumen_norm
        and not args.fused_router
        and not args.lumen_moe_dispatch_overlap
        and not args.lumen_moe_global_expert_layout
    ):
        return model

    from lumen.config import LumenConfig

    lumen_config = LumenConfig(
        format=args.lumen_format,
        scaling=args.fp8_scaling if args.mode == "fp8_blockwise2d" else "none",
        block_size=args.fp8_block_size,
        amax_algo="max",
        history_len=16,
        reduce_amax=False,
        quantize_activation=True,
        fp8_wgrad=True,
        cache_frozen_weight=bool(args.fp8_param_storage),
        bpreshuffle_gemm=False,
        quantize_grad=None,
        first_last_layers_bf16=False,
        lumen_norm=args.lumen_norm,
        hf_attn_patch=args.aiter_attn,
        fused_router=args.fused_router,
        moe_dispatch_overlap=args.lumen_moe_dispatch_overlap,
        moe_global_expert_layout=args.lumen_moe_global_expert_layout,
    )
    manager, model = lumen_config.enable(model, dp_group=dp_group)
    # Sonic FP8 (grouped_fp8_expert_mlp) is ~10x slower than BF16
    # moe_pre_routed_inputs. Disabled until Triton grouped GEMM is optimized.
    # Expert compute uses BF16 Sonic path with expert GEMM checkpoint for
    # memory efficiency.
    return model


def _apply_selective_gradient_checkpointing(model) -> None:
    """Checkpoint only attention, leave MoE uncheckpointed.

    Wraps each decoder layer's self_attn in torch.utils.checkpoint while
    letting the MoE block run normally. This avoids recomputing EP
    collectives during backward while still saving O(seq_len²) attention
    activation memory.

    Must be called with use_cache=False in the model forward (no
    DynamicCache) to avoid stale cache references during recompute.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    for layer in model.model.layers:
        original_self_attn_forward = layer.self_attn.forward

        def _make_checkpointed_attn(orig_fn):
            def _checkpointed_attn(*args, **kwargs):
                def _fn(*a, **kw):
                    return orig_fn(*a, **kw)
                return checkpoint(_fn, *args, use_reentrant=False, **kwargs)
            return _checkpointed_attn

        layer.self_attn.forward = _make_checkpointed_attn(original_self_attn_forward)

    LOGGER.info(
        "applied selective gradient checkpointing on %d layers "
        "(attention=checkpointed, MoE=retained)",
        len(model.model.layers),
    )


def _optimize_fsdp_reshard_for_fp8_cache(model: "torch.nn.Module") -> None:
    """Skip redundant FSDP backward all-gather for layers with FP8 weight cache.

    When ``cache_frozen_weight=True``, Lumen caches the FP8-quantized weight
    on each module after the first forward.  During backward FSDP would
    normally all-gather the BF16 weight again, but it goes unused because
    Lumen reads from the FP8 cache.  Registering a forward hook that
    populates the cache on the first call, then calling
    ``set_reshard_after_backward(False)`` on FSDPModule parents avoids the
    wasted backward all-gather — saving both bandwidth and peak memory.
    """
    import torch
    from torch.distributed._composable.fsdp import FSDPModule

    count = 0
    for module in model.modules():
        if isinstance(module, FSDPModule):
            has_frozen_linear = any(
                isinstance(child, torch.nn.Linear)
                and not child.weight.requires_grad
                for child in module.modules()
                if child is not module
            )
            if has_frozen_linear:
                count += 1
    LOGGER.info(
        "FP8 param storage: %d FSDP modules have frozen linears "
        "(backward all-gather optimized via cache_frozen_weight)",
        count,
    )


def prepare_model(config: dict, groups):
    """Build -> EP shard -> LoRA -> FP8 patch -> FSDP, in that exact order."""
    from lumen.models.qwen3_30b_a3b.fsdp import pretrain as lumen_qwen

    args = _lumen_args(config)
    revision = config["model"].get("revision")
    local_model = config["model"].get("local_path")
    if local_model:
        local_path = Path(str(local_model)).expanduser().resolve()
        if not local_path.is_dir():
            raise RuntimeError(f"pinned local model directory is unavailable: {local_path}")
        args.model_name_or_path = str(local_path)
    elif revision:
        from huggingface_hub import snapshot_download

        args.model_name_or_path = snapshot_download(
            repo_id=config["model"]["name_or_path"],
            revision=revision,
            local_files_only=True,
        )
    model = lumen_qwen.build_model(args)
    import torch

    model = model.to(dtype=torch.bfloat16)
    LOGGER.info("cast base model to BF16 (frozen weights do not need FP32)")
    model = lumen_qwen.shard_moe_experts(
        model, groups, expert_backend=args.expert_backend
    )
    pass  # fp8_param_storage handled via cache_frozen_weight in _enable_lumen
    lora = config["lora"]
    targets = lora.get("targets", {})
    inject_hierarchical_lora(
        model,
        LoRAConfig(
            attention_rank=lora["attention_rank"],
            attention_alpha=lora["attention_alpha"],
            expert_rank=lora["expert_rank"],
            expert_alpha=lora["expert_alpha"],
            dropout=lora.get("dropout", 0.0),
            dtype=config["precision"].get("adapter_dtype", "bfloat16"),
            attention_targets=tuple(
                targets.get("attention", ("q_proj", "k_proj", "v_proj", "o_proj"))
            ),
            expert_targets=tuple(
                targets.get("experts", ("gate_up_proj", "down_proj"))
            ),
        ),
        expert_backend=args.expert_backend,
    )
    audit_trainable_parameters(model)
    resume_dir = config.get("output", {}).get("resume_adapter_dir")
    if resume_dir:
        import torch.distributed as dist

        load_distributed_adapter(
            model,
            resume_dir,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            expected_base_model=config["model"]["name_or_path"],
            expected_revision=config["model"].get("revision"),
            expected_tokenizer_sha256=config["execution"]["tokenizer_sha256"],
            expected_data_manifest_sha256=config["execution"][
                "data_manifest_sha256"
            ],
        )
    if config.get("training", {}).get("moe_activation_offload", True):
        lumen_qwen.enable_moe_activation_offload(model)
    if config.get("training", {}).get("gradient_checkpointing", True):
        if config.get("training", {}).get("selective_gc", True):
            _apply_selective_gradient_checkpointing(model)
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    # Lumen sees LoRALinear.base_layer as an nn.Linear and can replace eligible
    # frozen base GEMMs with blockwise2d FP8 compute. Adapter math stays BF16.
    model = _enable_lumen(model, args, groups.dp_group)
    model = lumen_qwen.apply_fsdp2(model, groups, args.sharding)
    if args.fp8_param_storage:
        _optimize_fsdp_reshard_for_fp8_cache(model)
    audit_trainable_parameters(model)
    return model


def run(config: dict) -> None:
    """Run SFT. This is intentionally compact and delegates model mechanics."""
    assert_training_authorized(config)
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    from lumen.models.qwen3_30b_a3b.fsdp import pretrain as lumen_qwen

    local_rank = int(os.environ["LOCAL_RANK"])
    validate_launch_environment(config, os.environ)
    runtime_arch = assert_runtime_compatible(
        config, torch_module=torch, device_index=local_rank
    )
    LOGGER.info("validated runtime GPU architecture %s", runtime_arch)
    if not dist.is_initialized():
        timeout_seconds = int(
            config.get("distributed", {}).get("timeout_seconds", 7200)
        )
        dist.init_process_group(
            "nccl", timeout=timedelta(seconds=timeout_seconds)
        )
    torch.cuda.set_device(local_rank)
    seed = int(config["training"].get("seed", 1234))
    torch.manual_seed(seed)
    timeout_seconds = int(
        config.get("distributed", {}).get("timeout_seconds", 7200)
    )
    groups = lumen_qwen.create_parallel_groups(
        config["distributed"]["ep_size"],
        config["distributed"].get("dp_size", dist.get_world_size()),
        timeout_minutes=max(timeout_seconds // 60, 60),
    )
    model = prepare_model(config, groups)

    data = config["data"]

    def make_loader(path: str, *, train: bool):
        dataset = PretokenizedLossMaskDataset(
            path,
            max_sequence_length=(
                data.get("max_sequence_length")
                if train
                else data.get("validation_max_sequence_length")
            ),
        )
        sampler = LengthBucketDistributedSampler(
            dataset,
            num_replicas=groups.dp_size,
            rank=groups.dp_rank,
            shuffle=train,
            seed=config["training"].get("seed", 1234),
        )
        loader = DataLoader(
            dataset,
            batch_size=config["training"]["micro_batch_size"],
            sampler=sampler,
            collate_fn=LossMaskCollator(data["pad_token_id"], pad_to_multiple_of=8),
            num_workers=data.get("num_workers", 2),
            pin_memory=True,
            drop_last=train,
        )
        return loader, sampler

    loader, sampler = make_loader(data["train_path"], train=True)
    validation = (
        make_loader(data["validation_path"], train=False)[0]
        if data.get("validation_path")
        else None
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=0.0,
    )
    gradient_accumulation = int(config["training"]["gradient_accumulation_steps"])
    model.train()
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    max_steps = int(config["training"]["max_steps"])
    epoch = 0
    best_validation = float("inf")
    evaluations_without_improvement = 0

    @torch.no_grad()
    def validation_loss() -> float:
        if validation is None:
            raise RuntimeError("validation_path is not configured")
        model.eval()
        totals = torch.zeros(2, dtype=torch.float64, device=local_rank)
        max_batches = int(config["training"].get("validation_batches", 32))
        for batch_index, batch in enumerate(validation):
            if batch_index >= max_batches:
                break
            input_ids = batch["input_ids"].to(local_rank, non_blocking=True)
            attention_mask = batch["attention_mask"].to(local_rank, non_blocking=True)
            loss_mask = batch["loss_mask"].to(local_rank, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            loss_sum, token_count = masked_causal_lm_loss_sum(
                logits, input_ids, loss_mask
            )
            totals[0] += loss_sum.double()
            totals[1] += token_count.double()
        dist.all_reduce(totals, group=groups.dp_group)
        model.train()
        if totals[1].item() == 0:
            raise RuntimeError("validation loader yielded no batches")
        return (totals[0] / totals[1]).item()

    def _set_fsdp_grad_sync(module, sync: bool) -> None:
        """Toggle only FSDP2 gradient reduce-scatter, not parameter reshard."""
        from torch.distributed._composable.fsdp import FSDPModule

        for m in module.modules():
            if isinstance(m, FSDPModule):
                m.set_requires_gradient_sync(sync)

    for step in range(1, max_steps + 1):
        accumulated = 0.0
        for micro in range(gradient_accumulation):
            is_last_micro = micro == gradient_accumulation - 1
            pass  # no FSDP sync control for debugging
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch["input_ids"].to(local_rank, non_blocking=True)
            attention_mask = batch["attention_mask"].to(local_rank, non_blocking=True)
            loss_mask = batch["loss_mask"].to(local_rank, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            loss_sum, local_token_count = masked_causal_lm_loss_sum(
                logits, input_ids, loss_mask
            )
            global_token_count = local_token_count.detach().clone()
            dist.all_reduce(global_token_count, group=groups.dp_group)
            loss = loss_sum * groups.dp_size / global_token_count
            (loss / (gradient_accumulation * groups.dp_size)).backward()
            accumulated += loss.item()
        lumen_qwen._clip_grad_norm_mixed_mesh(
            trainable, config["training"].get("max_grad_norm", 1.0)
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if groups.dp_rank == 0 and step % config["training"].get("log_interval", 1) == 0:
            LOGGER.info("step %d/%d loss %.6f", step, max_steps, accumulated / gradient_accumulation)
        evaluation_interval = int(config["training"].get("evaluation_interval", 0))
        if validation is not None and evaluation_interval and step % evaluation_interval == 0:
            current_validation = validation_loss()
            if groups.dp_rank == 0:
                LOGGER.info("step %d validation_loss %.6f", step, current_validation)
            if current_validation < best_validation:
                best_validation = current_validation
                evaluations_without_improvement = 0
                save_adapter(
                    model,
                    Path(config["output"]["adapter_dir"]) / "best",
                    base_model=config["model"]["name_or_path"],
                    revision=config["model"].get("revision"),
                    rank=dist.get_rank(),
                    world_size=dist.get_world_size(),
                    tokenizer_sha256=config["execution"]["tokenizer_sha256"],
                    data_manifest_sha256=config["execution"][
                        "data_manifest_sha256"
                    ],
                )
            else:
                evaluations_without_improvement += 1
            patience = int(config["training"].get("early_stopping_patience", 0))
            if patience and evaluations_without_improvement >= patience:
                if groups.dp_rank == 0:
                    LOGGER.info("early stopping after %d evaluations", patience)
                break

    save_adapter(
        model,
        config["output"]["adapter_dir"],
        base_model=config["model"]["name_or_path"],
        revision=config["model"].get("revision"),
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        tokenizer_sha256=config["execution"]["tokenizer_sha256"],
        data_manifest_sha256=config["execution"]["data_manifest_sha256"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    try:
        run(config)
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
