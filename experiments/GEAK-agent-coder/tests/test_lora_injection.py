import pytest
import torch
from torch import nn

from geak_agent_coder.sft.lora import (
    LoRALinear,
    audit_trainable_parameters,
    inject_hierarchical_lora,
    validate_expert_backend,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.router = nn.Linear(8, 2, bias=False)
        self.local_experts = nn.Module()
        self.local_experts.experts = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "gate_up_proj": nn.Linear(8, 12, bias=False),
                        "down_proj": nn.Linear(6, 8, bias=False),
                    }
                )
                for _ in range(2)
            ]
        )
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 16, bias=False)


def test_hierarchical_lora_ranks_dtype_and_freezing():
    model = TinyModel()
    targets = inject_hierarchical_lora(model)
    audit = audit_trainable_parameters(model)

    assert len(targets) == 8
    assert isinstance(model.q_proj, LoRALinear)
    assert model.q_proj.rank == 32
    assert model.q_proj.alpha == 64
    assert model.local_experts.experts[0]["gate_up_proj"].rank == 8
    assert model.local_experts.experts[0]["gate_up_proj"].alpha == 16
    assert all(dict(model.named_parameters())[name].dtype == torch.bfloat16 for name in audit.trainable_names)
    assert not model.router.weight.requires_grad
    assert not model.embed_tokens.weight.requires_grad
    assert not model.norm.weight.requires_grad
    assert not model.lm_head.weight.requires_grad
    assert not model.q_proj.base_layer.weight.requires_grad


def test_expert_lora_rejects_te_grouped_backend():
    with pytest.raises(ValueError, match="requires expert_backend"):
        validate_expert_backend("te_grouped")


def test_sonic_packed_expert_lora_preserves_initial_weights():
    model = TinyModel()
    model.local_experts = nn.Module()
    model.local_experts.w1 = nn.Parameter(torch.randn(2, 8, 12))
    model.local_experts.w2 = nn.Parameter(torch.randn(2, 6, 8))

    targets = inject_hierarchical_lora(model, expert_backend="sonic")
    audit = audit_trainable_parameters(model)

    assert targets[-2:] == (
        "local_experts.gate_up_lora",
        "local_experts.down_lora",
    )
    assert torch.count_nonzero(model.local_experts.gate_up_lora.delta()) == 0
    assert torch.count_nonzero(model.local_experts.down_lora.delta()) == 0
    assert model.local_experts.gate_up_lora.alpha == 16
    assert audit.expert_parameters > 0
    assert not model.local_experts.w1.requires_grad
    assert not model.local_experts.w2.requires_grad


def test_lora_zero_initialization_preserves_output():
    model = TinyModel()
    reference = model.q_proj(torch.randn(3, 8))
    inputs = torch.randn(3, 8)
    expected = model.q_proj(inputs)
    inject_hierarchical_lora(model)
    torch.testing.assert_close(model.q_proj(inputs), expected)
    assert reference.shape == expected.shape
