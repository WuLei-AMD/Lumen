"""Tests for model-independent MoE dispatch overlap helpers."""

from argparse import Namespace

import pytest
import torch
import torch.nn as nn

from lumen.config import LumenConfig
from lumen.ops.moe.dispatch_layout import transpose_variable_chunks
from lumen.ops.moe.dispatch_overlap import begin_count_exchange


def test_variable_chunk_layout_round_trip_preserves_gradients():
    counts = [[2, 1, 0], [1, 2, 2]]
    sender_major = torch.arange(8, dtype=torch.float32, requires_grad=True)

    expert_major = transpose_variable_chunks(
        sender_major,
        counts,
        source_layout="sender_major",
    )
    restored = transpose_variable_chunks(
        expert_major,
        counts,
        source_layout="expert_major",
    )

    torch.testing.assert_close(restored, sender_major)
    restored.square().sum().backward()
    torch.testing.assert_close(sender_major.grad, 2 * sender_major.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_fused_variable_chunk_layout_matches_torch_fallback():
    pytest.importorskip("transformer_engine.pytorch.permutation")
    counts = torch.tensor([[2, 1, 0], [1, 2, 2]], device="cuda")
    sender_major = torch.arange(
        32,
        dtype=torch.float32,
        device="cuda",
    ).view(8, 4)

    expected = transpose_variable_chunks(
        sender_major,
        counts,
        source_layout="sender_major",
        fused=False,
    )
    actual = transpose_variable_chunks(
        sender_major,
        counts,
        source_layout="sender_major",
        fused=True,
    )
    restored = transpose_variable_chunks(
        actual,
        counts,
        source_layout="expert_major",
        fused=True,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(restored, sender_major)


def test_begin_count_exchange_single_rank_returns_local_splits():
    counts = torch.tensor([3, 5], dtype=torch.int64)

    exchange = begin_count_exchange(counts, group=None)

    assert exchange.wait_for_splits() == ([3, 5], [3, 5])


def test_begin_count_exchange_launches_async_collective(monkeypatch):
    counts = torch.tensor([3, 5], dtype=torch.int64)
    waited = []
    calls = []

    class _Work:
        def wait(self):
            waited.append(True)

    def _all_to_all_single(output, input_, *, group, async_op):
        calls.append((group, async_op))
        output.copy_(torch.tensor([7, 1]))
        return _Work()

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", _all_to_all_single)

    exchange = begin_count_exchange(counts, group="ep")
    assert calls == [("ep", True)]
    assert waited == []
    assert exchange.wait_for_splits() == ([3, 5], [7, 1])
    assert waited == [True]


def test_lumen_config_maps_and_patches_moe_dispatch_overlap():
    class _Dispatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.enabled = False

        def enable_lumen_moe_dispatch_overlap(self):
            self.enabled = True

    model = nn.Sequential(_Dispatcher())
    config = LumenConfig.from_args(
        Namespace(linear_fp8=False, lumen_moe_dispatch_overlap=True)
    )

    assert config.moe_dispatch_overlap is True
    assert config.has_any_features
    config.enable(model)
    assert model[0].enabled is True


def test_lumen_config_maps_and_patches_global_expert_layout():
    class _Dispatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.enabled = False

        def enable_lumen_moe_global_expert_layout(self):
            self.enabled = True

    model = nn.Sequential(_Dispatcher())
    config = LumenConfig.from_args(
        Namespace(linear_fp8=False, lumen_moe_global_expert_layout=True)
    )

    config.enable(model)
    assert config.moe_global_expert_layout is True
    assert model[0].enabled is True
