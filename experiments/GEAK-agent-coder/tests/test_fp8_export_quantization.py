import pytest
import torch

from geak_agent_coder.export.fp8 import (
    dequantize_blockwise_2d,
    quantize_blockwise_2d,
    scale_key_for,
    weight_eligibility,
)


def test_cpu_blockwise_fp8_round_trip_and_scale_shape():
    torch.manual_seed(4)
    weight = torch.randn(256, 128, dtype=torch.bfloat16)
    quantized, scale_inv = quantize_blockwise_2d(weight)
    restored = dequantize_blockwise_2d(quantized, scale_inv)

    assert quantized.dtype == torch.float8_e4m3fn
    assert scale_inv.shape == (2, 1)
    torch.testing.assert_close(restored, weight.float(), rtol=0.08, atol=0.03)


def test_zero_block_is_finite():
    weight = torch.zeros(128, 128)
    quantized, scale_inv = quantize_blockwise_2d(weight)
    restored = dequantize_blockwise_2d(quantized, scale_inv)
    assert torch.isfinite(scale_inv).all()
    assert torch.count_nonzero(restored) == 0


def test_packed_expert_stack_quantization():
    weight = torch.randn(3, 128, 256)
    quantized, scale_inv = quantize_blockwise_2d(weight)
    restored = dequantize_blockwise_2d(quantized, scale_inv)
    assert scale_inv.shape == (3, 1, 2)
    torch.testing.assert_close(restored, weight, rtol=0.08, atol=0.03)


@pytest.mark.parametrize(
    ("name", "shape", "eligible"),
    [
        ("model.layers.0.self_attn.q_proj.weight", (128, 256), True),
        ("model.embed_tokens.weight", (128, 256), False),
        ("model.layers.0.mlp.gate.weight", (128, 256), False),
        ("model.layers.0.mlp.experts.gate_up_proj", (4, 128, 256), True),
        ("model.layers.0.mlp.experts.gate_up_proj", (4, 129, 256), False),
        ("lm_head.weight", (128, 256), False),
        ("model.layers.0.self_attn.q_proj.weight", (129, 256), False),
    ],
)
def test_weight_eligibility_policy(name, shape, eligible):
    result = weight_eligibility(name, torch.ones(shape))
    assert result.eligible is eligible


def test_scale_key_matches_hf_convention():
    assert scale_key_for("model.q_proj.weight") == "model.q_proj.weight_scale_inv"
    assert (
        scale_key_for("model.layers.0.mlp.experts.gate_up_proj")
        == "model.layers.0.mlp.experts.gate_up_proj_scale_inv"
    )
