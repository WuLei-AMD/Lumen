"""MCore schema for Qwen3-MoE fields missing from the bundled converter."""

from schema_base import ModelSchema


def get_qwen3_moe_schema(num_local_experts: int) -> ModelSchema:
    layer = {
        "self_attn_norm_weight": "self_attention.linear_qkv.layer_norm_weight",
        "self_attn_qkv_weight": "self_attention.linear_qkv.weight",
        "self_attn_proj_weight": "self_attention.linear_proj.weight",
        "q_norm_weight": "self_attention.q_layernorm.weight",
        "k_norm_weight": "self_attention.k_layernorm.weight",
        "mlp_norm_weight": "pre_mlp_layernorm.weight",
        "router_weight": "mlp.router.weight",
    }
    for expert_idx in range(num_local_experts):
        layer[f"mlp_fc1_weight.{expert_idx}"] = (
            f"mlp.experts.local_experts.{expert_idx}.linear_fc1.weight"
        )
        layer[f"mlp_fc2_weight.{expert_idx}"] = (
            f"mlp.experts.local_experts.{expert_idx}.linear_fc2.weight"
        )
    return ModelSchema(
        {
            "embeddings": {
                "pos": "embedding.position_embeddings.weight",
                "word": "embedding.word_embeddings.weight",
            },
            "layer_prefix": "decoder.layers",
            "layer": layer,
            "final_norm": {
                "weight": "decoder.final_layernorm.weight",
                "bias": "decoder.final_layernorm.bias",
            },
            "output_layer": {"weight": "output_layer.weight"},
        }
    )
