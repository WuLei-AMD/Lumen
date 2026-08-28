import pathlib
import sys

import pytest
import torch

CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHECKPOINT_DIR))

from qwen3_moe_mapping import (  # noqa: E402
    expert_range_for_ep_rank,
    pack_grouped_query_qkv,
    pack_swiglu_fc1,
)


def test_qkv_is_packed_per_query_group():
    query = torch.arange(8 * 3).reshape(8, 3)
    key = 100 + torch.arange(4 * 3).reshape(4, 3)
    value = 200 + torch.arange(4 * 3).reshape(4, 3)

    packed = pack_grouped_query_qkv(
        query,
        key,
        value,
        num_attention_heads=4,
        num_query_groups=2,
        head_dim=2,
    )

    expected = torch.cat(
        (
            query[:4],
            key[:2],
            value[:2],
            query[4:],
            key[2:],
            value[2:],
        )
    )
    torch.testing.assert_close(packed, expected)


def test_swiglu_pack_preserves_gate_then_up_for_each_expert():
    gate = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    up = 100 + gate
    packed = pack_swiglu_fc1(gate, up)

    assert packed.shape == (2, 6, 4)
    torch.testing.assert_close(packed[:, :3], gate)
    torch.testing.assert_close(packed[:, 3:], up)


def test_ep8_owns_contiguous_sixteen_expert_shards():
    ranges = [expert_range_for_ep_rank(128, 8, rank) for rank in range(8)]
    assert [len(owned) for owned in ranges] == [16] * 8
    assert [index for owned in ranges for index in owned] == list(range(128))
    assert list(ranges[7]) == list(range(112, 128))


def test_invalid_expert_partition_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        expert_range_for_ep_rank(10, 8, 0)
