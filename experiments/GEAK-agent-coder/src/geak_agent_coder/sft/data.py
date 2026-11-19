"""Pretokenized, answer-masked SFT data loading."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator


class PretokenizedLossMaskDataset:
    """JSONL dataset containing aligned ``input_ids`` and ``loss_mask`` lists."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_samples: int | None = None,
        max_sequence_length: int | None = None,
    ):
        self.path = Path(path)
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        with self.path.open("rb") as handle:
            while max_samples is None or len(self.offsets) < max_samples:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    row = json.loads(line)
                    input_ids = row.get("input_ids")
                    if not isinstance(input_ids, list):
                        raise ValueError(
                            f"row {len(self.offsets)}: input_ids must be a list"
                        )
                    if (
                        max_sequence_length is not None
                        and len(input_ids) > max_sequence_length
                    ):
                        continue
                    self.offsets.append(offset)
                    self.lengths.append(len(input_ids))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            row = json.loads(handle.readline())
        return validate_pretokenized_row(row, index=index)


class LengthBucketDistributedSampler:
    """Assign similarly sized sequences to ranks in each synchronous step."""

    def __init__(
        self,
        dataset: PretokenizedLossMaskDataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(dataset) // num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        ordered = sorted(
            range(len(self.dataset)),
            key=lambda index: (self.dataset.lengths[index], index),
        )
        usable = self.num_samples * self.num_replicas
        ordered = ordered[:usable]
        groups = [
            ordered[start : start + self.num_replicas]
            for start in range(0, usable, self.num_replicas)
        ]
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(groups)
        return iter(group[self.rank] for group in groups)


def validate_pretokenized_row(
    row: dict[str, Any], *, index: int | None = None
) -> dict[str, list[int]]:
    location = f"row {index}" if index is not None else "row"
    ids = row.get("input_ids")
    mask = row.get("loss_mask")
    if not isinstance(ids, list) or not isinstance(mask, list):
        raise ValueError(f"{location}: input_ids and loss_mask must be lists")
    if len(ids) != len(mask) or len(ids) < 2:
        raise ValueError(f"{location}: input_ids/loss_mask must align and contain >=2 tokens")
    if not all(isinstance(token, int) and token >= 0 for token in ids):
        raise ValueError(f"{location}: input_ids must contain non-negative integers")
    if not all(value in (0, 1, False, True) for value in mask):
        raise ValueError(f"{location}: loss_mask must be binary")
    if not any(mask[1:]):
        raise ValueError(f"{location}: shifted loss_mask selects no target tokens")
    return {"input_ids": ids, "loss_mask": [int(value) for value in mask]}


class LossMaskCollator:
    """Pad pretokenized rows without turning padding into loss targets."""

    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int | None = None):
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, rows: Iterable[dict[str, list[int]]]):
        import torch

        rows = list(rows)
        if not rows:
            raise ValueError("cannot collate an empty batch")
        validated = [validate_pretokenized_row(row) for row in rows]
        width = max(len(row["input_ids"]) for row in validated)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            width = ((width + multiple - 1) // multiple) * multiple
        input_ids, loss_mask, attention_mask = [], [], []
        for row in validated:
            length = len(row["input_ids"])
            padding = width - length
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            loss_mask.append(row["loss_mask"] + [0] * padding)
            attention_mask.append([1] * length + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        }


def masked_causal_lm_loss_sum(logits, input_ids, loss_mask):
    """Return selected next-token loss sum and selected-token count."""
    import torch.nn.functional as functional

    if input_ids.shape != loss_mask.shape:
        raise ValueError("input_ids and loss_mask shapes must match")
    labels = input_ids[:, 1:]
    selected = loss_mask[:, 1:].float()
    token_loss = functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    )
    denominator = selected.sum()
    if denominator.item() == 0:
        raise ValueError("batch loss_mask selects no target tokens")
    return (token_loss.float() * selected.reshape(-1)).sum(), denominator


def masked_causal_lm_loss(logits, input_ids, loss_mask):
    """Compute next-token cross entropy over selected assistant tokens."""

    loss_sum, token_count = masked_causal_lm_loss_sum(
        logits, input_ids, loss_mask
    )
    return loss_sum / token_count
