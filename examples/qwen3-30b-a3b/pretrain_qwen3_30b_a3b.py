"""Qwen3-30B-A3B Megatron pretraining entry point."""

import functools
import os
from pathlib import Path

from megatron.core.enums import ModelType
from megatron.training import pretrain

from lumen.models.megatron import (
    install_fp8_param_gather_hook,
    make_lumen_model_provider,
)
from lumen.models.qwen3_30b_a3b.megatron import (
    add_pretrain_args,
    apply_fp8_training,
    apply_lora,
    forward_step,
    lumen_gpt_builder,
    train_valid_test_datasets_provider,
)


model_provider = make_lumen_model_provider(
    lumen_gpt_builder,
    lora_applier=apply_lora,
    fp8_applier=apply_fp8_training,
)
install_fp8_param_gather_hook()
train_valid_test_datasets_provider.is_distributed = True


def _install_e2e_profiler() -> None:
    """Profile one or more complete Megatron train steps."""
    target_raw = os.environ.get("QWEN_E2E_PROFILE_STEP")
    if target_raw is None:
        return

    import torch
    import torch.distributed as dist
    import megatron.training.training as training

    target = int(target_raw)
    num_steps = int(os.environ.get("QWEN_E2E_PROFILE_STEPS", "1"))
    if num_steps < 1:
        raise ValueError("QWEN_E2E_PROFILE_STEPS must be at least 1")
    output_dir = Path(os.environ.get("QWEN_E2E_PROFILE_DIR", "/tmp/qwen-e2e-profile"))
    profile_all_ranks = os.environ.get("QWEN_E2E_PROFILE_ALL_RANKS", "0") == "1"
    state = {"step": 0}
    original_train_step = training.train_step

    @functools.wraps(original_train_step)
    def profiled_train_step(*args, **kwargs):
        state["step"] += 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        in_profile_range = target <= state["step"] < target + num_steps
        if not in_profile_range or (rank != 0 and not profile_all_ranks):
            return original_train_step(*args, **kwargs)

        output_dir.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=False,
            with_stack=False,
        ) as profiler:
            result = original_train_step(*args, **kwargs)

        averages = profiler.key_averages()
        tables = [
            "=== Self CUDA time ===",
            averages.table(sort_by="self_cuda_time_total", row_limit=200),
            "\n=== Total CUDA time ===",
            averages.table(sort_by="cuda_time_total", row_limit=200),
            "\n=== Self CPU time ===",
            averages.table(sort_by="self_cpu_time_total", row_limit=100),
        ]
        step_suffix = f"-step{state['step']}" if num_steps > 1 else ""
        (output_dir / f"rank{rank}{step_suffix}-operators.txt").write_text("\n".join(tables))
        if os.environ.get("QWEN_E2E_PROFILE_TRACE", "0") == "1":
            profiler.export_chrome_trace(
                str(output_dir / f"rank{rank}{step_suffix}-trace.json")
            )
        return result

    training.train_step = profiled_train_step


if __name__ == "__main__":
    _install_e2e_profiler()
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_pretrain_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )
