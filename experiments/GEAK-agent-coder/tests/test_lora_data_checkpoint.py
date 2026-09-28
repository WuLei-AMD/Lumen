import json

import pytest
import torch

from geak_agent_coder.sft.checkpoint import extract_adapter_state
from geak_agent_coder.sft.data import (
    LengthBucketDistributedSampler,
    LossMaskCollator,
    PretokenizedLossMaskDataset,
    masked_causal_lm_loss,
)
from geak_agent_coder.sft.lora import inject_hierarchical_lora
from test_lora_injection import TinyModel


def test_pretokenized_dataset_and_collator(tmp_path):
    path = tmp_path / "tokens.jsonl"
    rows = [
        {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 1]},
        {"input_ids": [4, 5, 6, 7], "loss_mask": [0, 1, 1, 1]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    dataset = PretokenizedLossMaskDataset(path)
    batch = LossMaskCollator(0, pad_to_multiple_of=4)([dataset[0], dataset[1]])

    assert batch["input_ids"].shape == (2, 4)
    assert batch["loss_mask"][0].tolist() == [0, 0, 1, 0]
    assert batch["attention_mask"][0].tolist() == [True, True, True, False]

    logits = torch.randn(2, 4, 10)
    loss = masked_causal_lm_loss(logits, batch["input_ids"], batch["loss_mask"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dataset_rejects_empty_shifted_mask(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"input_ids":[1,2],"loss_mask":[1,0]}\n')
    dataset = PretokenizedLossMaskDataset(path)
    with pytest.raises(ValueError, match="selects no target"):
        dataset[0]


def test_length_bucket_sampler_balances_synchronous_rank_lengths(tmp_path):
    path = tmp_path / "tokens.jsonl"
    rows = [
        {"input_ids": list(range(length)), "loss_mask": [0] + [1] * (length - 1)}
        for length in (2, 3, 10, 11, 100, 101, 1000, 1001)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    dataset = PretokenizedLossMaskDataset(path)
    ranks = [
        list(
            LengthBucketDistributedSampler(
                dataset, num_replicas=2, rank=rank, shuffle=False, seed=0
            )
        )
        for rank in range(2)
    ]

    paired_lengths = [
        (dataset.lengths[ranks[0][step]], dataset.lengths[ranks[1][step]])
        for step in range(len(ranks[0]))
    ]
    assert paired_lengths == [(2, 3), (10, 11), (100, 101), (1000, 1001)]


def test_extract_adapter_state_only_returns_lora_tensors():
    model = TinyModel()
    inject_hierarchical_lora(model)
    state = extract_adapter_state(model)
    assert state
    assert all(name.endswith((".lora_A", ".lora_B")) for name in state)
    assert all(tensor.device.type == "cpu" for tensor in state.values())
