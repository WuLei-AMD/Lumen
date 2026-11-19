"""Qwen3-30B-A3B pretraining components for Megatron-LM-AMD."""

from lumen.models.llama31.megatron.pretrain import (
    forward_step,
    train_valid_test_datasets_provider,
)
from lumen.models.megatron import (
    add_common_megatron_args,
    apply_fp8_training,
    apply_lora,
)
from lumen.models.megatron import lumen_gpt_builder as _generic_gpt_builder
from lumen.models.utils import safe_add_argument

QWEN3_30B_A3B_CONFIG = {
    "num_layers": 48,
    "hidden_size": 2048,
    "ffn_hidden_size": 6144,
    "num_attention_heads": 32,
    "num_query_groups": 4,
    "kv_channels": 128,
    "num_experts": 128,
    "moe_ffn_hidden_size": 768,
    "moe_router_topk": 8,
    "vocab_size": 151936,
    "seq_length": 4096,
    "max_position_embeddings": 4096,
    "rotary_base": 1_000_000,
}


def lumen_gpt_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    """Build Qwen3-30B-A3B with the selected Lumen/MoE backend."""
    return _generic_gpt_builder(
        args,
        pre_process,
        post_process,
        vp_stage=vp_stage,
        config=config,
        pg_collection=pg_collection,
        model_name="Qwen3-30B-A3B",
    )


def add_pretrain_args(parser):
    """Register Qwen3/Lumen command-line options."""
    add_common_megatron_args(parser)
    safe_add_argument(
        parser,
        "--size",
        type=str,
        default="30b-a3b",
        choices=["30b-a3b"],
        help="Model size (Docker launcher compatibility).",
    )
    return parser


__all__ = [
    "QWEN3_30B_A3B_CONFIG",
    "add_pretrain_args",
    "apply_fp8_training",
    "apply_lora",
    "forward_step",
    "lumen_gpt_builder",
    "train_valid_test_datasets_provider",
]
