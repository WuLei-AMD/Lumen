###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Transformers + FSDP2 backend for Qwen3-30B-A3B."""

from lumen.models.qwen3_30b_a3b.fsdp.pretrain import (
    AlpacaDataset,
    EPShardedMoeBlock,
    FSDPTrainer,
    ParallelGroups,
    QWEN3_30B_A3B_CONFIG,
    apply_fsdp2,
    build_model,
    create_parallel_groups,
    get_args,
    main,
    shard_moe_experts,
)

__all__ = [
    "AlpacaDataset",
    "EPShardedMoeBlock",
    "FSDPTrainer",
    "ParallelGroups",
    "QWEN3_30B_A3B_CONFIG",
    "apply_fsdp2",
    "build_model",
    "create_parallel_groups",
    "get_args",
    "main",
    "shard_moe_experts",
]
