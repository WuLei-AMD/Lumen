"""Tests for the framework-agnostic pretraining dataset."""

import json
import importlib.util
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "lumen"
    / "models"
    / "llama31"
    / "dataset.py"
)
SPEC = importlib.util.spec_from_file_location("pretrain_dataset_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PretrainTextDataset = MODULE.PretrainTextDataset


class _Tokenizer:
    eos_token_id = 0

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(token) for token in text.split()]


def test_virtual_samples_cycle_physical_chunks(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(json.dumps({"text": "1 2 3 4 5 6 7 8 9 10 11"}) + "\n")

    dataset = PretrainTextDataset(
        str(data_path),
        seq_length=3,
        tokenizer=_Tokenizer(),
        is_hf_tokenizer=True,
        virtual_num_samples=5,
    )

    assert len(dataset) == 5
    assert torch.equal(dataset[3]["input_ids"], dataset[0]["input_ids"])
    assert torch.equal(dataset[4]["labels"], dataset[1]["labels"])


def test_virtual_samples_do_not_shrink_dataset(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(json.dumps({"text": "1 2 3 4 5 6 7 8 9 10 11"}) + "\n")

    dataset = PretrainTextDataset(
        str(data_path),
        seq_length=3,
        tokenizer=_Tokenizer(),
        is_hf_tokenizer=True,
        virtual_num_samples=2,
    )

    assert len(dataset) == 3
