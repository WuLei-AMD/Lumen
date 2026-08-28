"""CPU tests for the Qwen3-30B-A3B Transformers/FSDP implementation."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "lumen"
    / "models"
    / "qwen3_30b_a3b"
    / "fsdp"
    / "pretrain.py"
)
SPEC = importlib.util.spec_from_file_location("qwen3_fsdp_pretrain_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
EPShardedMoeBlock = MODULE.EPShardedMoeBlock
SonicLocalExperts = MODULE._SonicLocalExperts
create_parallel_groups = MODULE.create_parallel_groups


class _FakeGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.top_k = 2
        self.norm_topk_prob = True
        self.weight = nn.Parameter(torch.randn(4, 6))

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight)
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights.to(hidden_states.dtype), indices


class _FakeExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.randn(4, 10, 6))
        self.down_proj = nn.Parameter(torch.randn(4, 6, 5))
        self.act_fn = F.silu


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _FakeGate()
        self.experts = _FakeExperts()

    def forward(self, hidden_states):
        shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, shape[-1])
        _, weights, indices = self.gate(hidden_flat)
        output = torch.zeros_like(hidden_flat)
        for expert_id in range(self.experts.num_experts):
            token_ids, slots = torch.where(indices == expert_id)
            if token_ids.numel() == 0:
                continue
            gate, up = F.linear(
                hidden_flat[token_ids],
                self.experts.gate_up_proj[expert_id],
            ).chunk(2, dim=-1)
            expert_output = F.linear(
                F.silu(gate) * up,
                self.experts.down_proj[expert_id],
            )
            output.index_add_(
                0,
                token_ids,
                expert_output * weights[token_ids, slots].unsqueeze(-1),
            )
        return output.reshape(shape)


def test_ep_sharded_moe_ep1_matches_huggingface_layout():
    torch.manual_seed(7)
    reference = _FakeBlock()
    sharded_source = copy.deepcopy(reference)
    sharded = EPShardedMoeBlock(sharded_source, ep_rank=0, ep_size=1, ep_group=None)

    reference_input = torch.randn(2, 3, 6, requires_grad=True)
    sharded_input = reference_input.detach().clone().requires_grad_(True)
    reference_output = reference(reference_input)
    sharded_output = sharded(sharded_input)

    torch.testing.assert_close(sharded_output, reference_output)

    reference_output.square().sum().backward()
    sharded_output.square().sum().backward()
    torch.testing.assert_close(sharded_input.grad, reference_input.grad)
    torch.testing.assert_close(sharded.gate.weight.grad, reference.gate.weight.grad)
    sharded_gate_up_grad = torch.stack(
        [expert.gate_up_proj.weight.grad for expert in sharded.local_experts.experts]
    )
    sharded_down_grad = torch.stack(
        [expert.down_proj.weight.grad for expert in sharded.local_experts.experts]
    )
    torch.testing.assert_close(
        sharded_gate_up_grad,
        reference.experts.gate_up_proj.grad,
    )
    torch.testing.assert_close(
        sharded_down_grad,
        reference.experts.down_proj.grad,
    )


def test_sonic_blas_ep1_matches_huggingface_layout(monkeypatch):
    monkeypatch.setenv("SONIC_MOE_GEMM_BACKEND", "blas")
    torch.manual_seed(11)
    reference = _FakeBlock()
    sharded = EPShardedMoeBlock(
        copy.deepcopy(reference),
        ep_rank=0,
        ep_size=1,
        ep_group=None,
        expert_backend="sonic",
    )

    reference_input = torch.randn(2, 3, 6, requires_grad=True)
    sharded_input = reference_input.detach().clone().requires_grad_(True)
    reference_output = reference(reference_input)
    sharded_output = sharded(sharded_input)
    torch.testing.assert_close(sharded_output, reference_output)

    reference_output.square().sum().backward()
    sharded_output.square().sum().backward()
    torch.testing.assert_close(sharded_input.grad, reference_input.grad)
    reference_gate, reference_up = reference.experts.gate_up_proj.grad.chunk(2, dim=1)
    torch.testing.assert_close(
        sharded.local_experts.w1.grad,
        torch.stack((reference_gate, reference_up), dim=2).flatten(1, 2),
    )
    torch.testing.assert_close(
        sharded.local_experts.w2.grad,
        reference.experts.down_proj.grad,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_sonic_triton_forward_and_gradients(monkeypatch):
    pytest.importorskip("aiter.ops.triton.sonicmoe")
    monkeypatch.setenv("SONIC_MOE_GEMM_BACKEND", "triton")
    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    experts = _FakeExperts()
    experts.gate_up_proj = nn.Parameter(
        torch.randn(4, 128, 128, device=device, dtype=dtype) * 0.02
    )
    experts.down_proj = nn.Parameter(
        torch.randn(4, 128, 64, device=device, dtype=dtype) * 0.02
    )
    sonic = SonicLocalExperts(experts, 0, 4)

    hidden = torch.randn(19, 128, device=device, dtype=dtype, requires_grad=True)
    expert_ids = torch.tensor(
        [0, 2, 1, 3, 1, 0, 2, 2, 3, 0, 1, 3, 2, 0, 1, 1, 3, 2, 0],
        device=device,
    )
    weights = torch.rand(19, device=device, dtype=dtype, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weights = weights.detach().clone().requires_grad_(True)
    reference_w1 = sonic.w1.detach().clone().requires_grad_(True)
    reference_w2 = sonic.w2.detach().clone().requires_grad_(True)

    reference_output = torch.empty_like(reference_hidden)
    for expert_id in range(4):
        positions = torch.where(expert_ids == expert_id)[0]
        gate = F.linear(reference_hidden[positions], reference_w1[expert_id, 0::2])
        up = F.linear(reference_hidden[positions], reference_w1[expert_id, 1::2])
        result = F.linear(F.silu(gate) * up, reference_w2[expert_id])
        reference_output[positions] = result * reference_weights[positions, None]
    sonic_output = sonic.forward_all(hidden, expert_ids, weights)
    torch.testing.assert_close(sonic_output, reference_output, rtol=0.05, atol=0.02)

    output_gradient = torch.randn_like(sonic_output)
    sonic_output.backward(output_gradient)
    reference_output.backward(output_gradient)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.08, atol=0.03)
    torch.testing.assert_close(weights.grad, reference_weights.grad, rtol=0.08, atol=0.03)
    torch.testing.assert_close(sonic.w1.grad, reference_w1.grad, rtol=0.08, atol=0.03)
    torch.testing.assert_close(sonic.w2.grad, reference_w2.grad, rtol=0.08, atol=0.03)


def test_single_rank_parallel_groups_do_not_require_distributed_init():
    groups = create_parallel_groups(ep_size=1)
    assert groups.ep_group is None
    assert groups.dp_group is None
    assert groups.ep_rank == 0
    assert groups.dp_rank == 0
    assert groups.ep_size == 1
    assert groups.dp_size == 1
