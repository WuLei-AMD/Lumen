from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lumen.config import LumenConfig
from lumen.modules.sonic_moe import SonicMoEExperts


class _Linear(nn.Module):
    def __init__(self, out_features, in_features, value):
        super().__init__()
        self.weight = nn.Parameter(torch.full((out_features, in_features), value))


class _Expert(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.linear_fc1 = _Linear(8, 4, value)
        self.linear_fc2 = _Linear(4, 4, value + 10)


class _SequentialExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            add_bias_linear=False,
            gated_linear_unit=True,
            expert_tensor_parallel_size=1,
        )
        self.num_local_experts = 2
        self.local_experts = nn.ModuleList([_Expert(1.0), _Expert(2.0)])


class MoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _SequentialExperts()


def test_sonic_moe_packs_megatron_expert_weights():
    sonic = SonicMoEExperts(_SequentialExperts())

    assert sonic.w1.shape == (2, 8, 4)
    assert sonic.w2.shape == (2, 4, 4)
    assert sonic.w1.is_contiguous()
    assert sonic.w2.is_contiguous()
    torch.testing.assert_close(sonic.w1[0], torch.ones(8, 4))
    torch.testing.assert_close(sonic.w1[1], torch.full((8, 4), 2.0))
    torch.testing.assert_close(sonic.w2[0], torch.full((4, 4), 11.0))


def test_lumen_config_enable_replaces_megatron_moe():
    model = nn.Sequential(MoELayer())
    manager, returned = LumenConfig(scaling="none", sonic_moe=True).enable(model)

    assert manager is None
    assert returned is model
    assert isinstance(model[0].experts, SonicMoEExperts)


def test_sonic_blas_backend_uses_pre_routed_expert_blocks(monkeypatch):
    monkeypatch.setenv("SONIC_MOE_GEMM_BACKEND", "blas")
    sonic = SonicMoEExperts(_SequentialExperts())
    hidden = (torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10).requires_grad_()
    probs = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32, requires_grad=True)
    counts = torch.tensor([2, 1], dtype=torch.int64)

    output, bias = sonic(hidden, counts, probs)

    expected_chunks = []
    for expert_id, (tokens, expert_probs) in enumerate(
        zip(torch.split(hidden, [2, 1]), torch.split(probs, [2, 1]))
    ):
        intermediate = torch.nn.functional.linear(tokens, sonic.w1[expert_id])
        gate, up = torch.chunk(intermediate, 2, dim=-1)
        activated = (
            torch.nn.functional.silu(gate)
            * up
            * expert_probs.unsqueeze(-1)
        )
        expert_output = torch.nn.functional.linear(
            activated, sonic.w2[expert_id]
        )
        expected_chunks.append(expert_output)

    torch.testing.assert_close(output, torch.cat(expected_chunks))
    grad_output = torch.randn_like(output)
    actual_grads = torch.autograd.grad(
        output,
        (hidden, sonic.w1, sonic.w2, probs),
        grad_output,
        retain_graph=True,
    )
    expected_grads = torch.autograd.grad(
        torch.cat(expected_chunks),
        (hidden, sonic.w1, sonic.w2, probs),
        grad_output,
    )
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)
    assert bias is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_sonic_pre_routed_matches_general_routing():
    from aiter.ops.triton.sonicmoe import (
        SonicMoEActivationType,
        moe_general_routing_inputs,
        moe_pre_routed_inputs,
    )

    torch.manual_seed(123)
    device = torch.device("cuda")
    num_tokens, hidden_size, intermediate_size, num_experts = 64, 64, 64, 2
    counts = torch.tensor([32, 32], dtype=torch.int32, device=device)
    expert_indices = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device), counts
    )
    token_indices = torch.arange(num_tokens, dtype=torch.int32, device=device)

    x_general = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    x_pre_routed = x_general.detach().clone().requires_grad_(True)
    scores_general = torch.rand(num_tokens, dtype=torch.float32, device=device, requires_grad=True)
    scores_pre_routed = scores_general.detach().clone().requires_grad_(True)

    w1_general_storage = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w2_general_storage = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w1_pre_routed_storage = w1_general_storage.detach().clone().requires_grad_(True)
    w2_pre_routed_storage = w2_general_storage.detach().clone().requires_grad_(True)
    w1_general = w1_general_storage.permute(1, 2, 0)
    w2_general = w2_general_storage.permute(1, 2, 0)
    w1_pre_routed = w1_pre_routed_storage.permute(1, 2, 0)
    w2_pre_routed = w2_pre_routed_storage.permute(1, 2, 0)

    output_general, _ = moe_general_routing_inputs(
        x_general,
        scores_general,
        token_indices,
        expert_indices,
        w1_general,
        None,
        w2_general,
        None,
        num_experts,
        torch.cuda.current_stream().cuda_stream,
        SonicMoEActivationType.SWIGLU,
        False,
        True,
    )
    output_pre_routed, _ = moe_pre_routed_inputs(
        x_pre_routed,
        scores_pre_routed,
        counts,
        w1_pre_routed,
        None,
        w2_pre_routed,
        None,
        torch.cuda.current_stream().cuda_stream,
        SonicMoEActivationType.SWIGLU,
        False,
        True,
    )

    torch.testing.assert_close(output_pre_routed, output_general)
    grad = torch.randn_like(output_general)
    output_general.backward(grad)
    output_pre_routed.backward(grad)
    torch.testing.assert_close(x_pre_routed.grad, x_general.grad)
    torch.testing.assert_close(scores_pre_routed.grad, scores_general.grad)
    torch.testing.assert_close(w1_pre_routed_storage.grad, w1_general_storage.grad)
    torch.testing.assert_close(w2_pre_routed_storage.grad, w2_general_storage.grad)
