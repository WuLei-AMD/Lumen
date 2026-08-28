"""Qwen3-30B-A3B training components.

Backends:

- ``lumen.models.qwen3_30b_a3b.megatron`` for Megatron pretraining.
- ``lumen.models.qwen3_30b_a3b.fsdp`` for Transformers + FSDP2 DP x EP
  full-parameter training.

Megatron APIs remain re-exported here for backward compatibility.
"""

from lumen.models.qwen3_30b_a3b.megatron import (
    QWEN3_30B_A3B_CONFIG,
    add_pretrain_args,
    apply_fp8_training,
    apply_lora,
    forward_step,
    lumen_gpt_builder,
    train_valid_test_datasets_provider,
)

__all__ = [
    "QWEN3_30B_A3B_CONFIG",
    "add_pretrain_args",
    "apply_fp8_training",
    "apply_lora",
    "forward_step",
    "lumen_gpt_builder",
    "train_valid_test_datasets_provider",
]
