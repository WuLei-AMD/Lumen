"""CPU tests for the Qwen3-30B-A3B Transformers/FSDP implementation."""

import copy
import importlib.util
import sys
from pathlib import Path

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


def test_single_rank_parallel_groups_do_not_require_distributed_init():
    groups = create_parallel_groups(ep_size=1)
    assert groups.ep_group is None
    assert groups.dp_group is None
    assert groups.ep_rank == 0
    assert groups.dp_rank == 0
    assert groups.ep_size == 1
    assert groups.dp_size == 1
