"""Deterministic length-aware sampling by assistant loss-token share."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .contracts import DataContractError
from .formatting import TokenizedSample


@dataclass(frozen=True)
class SamplingConfig:
    replay_min_share: float = 0.15
    replay_max_share: float = 0.20
    seed: int = 0
    length_penalty_power: float = 0.5
    replay_mix_enabled: bool = True

    def __post_init__(self) -> None:
        if self.replay_mix_enabled and not (
            0 < self.replay_min_share <= self.replay_max_share < 1
        ):
            raise ValueError("replay share bounds must satisfy 0 < min <= max < 1")
        if self.length_penalty_power < 0:
            raise ValueError("length_penalty_power must be non-negative")


@dataclass(frozen=True)
class SamplingPlan:
    indices: tuple[int, ...]
    replay_loss_tokens: int
    total_loss_tokens: int
    replay_share: float
    sample_counts: dict[str, int]
    loss_tokens: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "indices": list(self.indices),
            "replay_loss_tokens": self.replay_loss_tokens,
            "total_loss_tokens": self.total_loss_tokens,
            "replay_share": self.replay_share,
            "sample_counts": dict(self.sample_counts),
            "loss_tokens": dict(self.loss_tokens),
        }


def _is_replay(sample: TokenizedSample) -> bool:
    return sample.sample_domain == "general_coding"


def _stable_rank(
    sample: TokenizedSample, index: int, config: SamplingConfig
) -> tuple[float, str, int]:
    digest = hashlib.sha256(
        f"{config.seed}:{sample.sample_id}".encode("utf-8")
    ).hexdigest()
    penalty = math.pow(max(len(sample.input_ids), 1), config.length_penalty_power)
    # Seeded jitter resolves ties while length remains the primary weighting.
    jitter = int(digest[:12], 16) / float(16**12)
    return (penalty * (1.0 + jitter * 0.01), digest, index)


def _stratum(sample: TokenizedSample) -> tuple[str, ...]:
    if _is_replay(sample):
        return (
            "general_coding",
            sample.primary_language or "<unknown>",
            sample.task_type or "<unknown>",
        )
    return (
        "kernel",
        sample.lane or "<unknown>",
        sample.task_type or "<unknown>",
        sample.implementation_family_id or "<unknown>",
    )


def build_sampling_plan(
    samples: Sequence[TokenizedSample],
    *,
    steps: int,
    config: SamplingConfig = SamplingConfig(),
) -> SamplingPlan:
    """Build a repeatable epoch schedule with a bounded replay token share.

    Selection is greedy over all domain candidates. It minimizes distance to
    the midpoint target, then prefers shorter examples using a deterministic
    seeded rank. Samples cycle within each domain, so no random library or
    distributed-process state can alter the schedule.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if not samples:
        raise DataContractError("cannot sample an empty dataset")
    if any(sample.assistant_loss_tokens <= 0 for sample in samples):
        raise DataContractError("every sample must have positive loss tokens")
    replay = [index for index, sample in enumerate(samples) if _is_replay(sample)]
    kernel = [index for index, sample in enumerate(samples) if not _is_replay(sample)]
    if not config.replay_mix_enabled:
        ordered = sorted(
            range(len(samples)),
            key=lambda index: _stable_rank(samples[index], index, config),
        )
        uses = {index: 0 for index in ordered}
        chosen: list[int] = []
        for _ in range(steps):
            index = min(
                ordered,
                key=lambda candidate: (
                    (uses[candidate] + 1)
                    * _stable_rank(samples[candidate], candidate, config)[0],
                    _stable_rank(samples[candidate], candidate, config)[1],
                ),
            )
            chosen.append(index)
            uses[index] += 1
        replay_tokens = sum(
            samples[index].assistant_loss_tokens
            for index in chosen
            if _is_replay(samples[index])
        )
        total_tokens = sum(
            samples[index].assistant_loss_tokens for index in chosen
        )
        sample_counts: defaultdict[str, int] = defaultdict(int)
        loss_tokens: defaultdict[str, int] = defaultdict(int)
        for index in chosen:
            domain = samples[index].sample_domain
            sample_counts[domain] += 1
            loss_tokens[domain] += samples[index].assistant_loss_tokens
        return SamplingPlan(
            indices=tuple(chosen),
            replay_loss_tokens=replay_tokens,
            total_loss_tokens=total_tokens,
            replay_share=replay_tokens / total_tokens,
            sample_counts=dict(sorted(sample_counts.items())),
            loss_tokens=dict(sorted(loss_tokens.items())),
        )
    if not replay or not kernel:
        raise DataContractError("sampler requires both kernel and replay samples")
    replay.sort(key=lambda index: _stable_rank(samples[index], index, config))
    kernel.sort(key=lambda index: _stable_rank(samples[index], index, config))

    midpoint = (config.replay_min_share + config.replay_max_share) / 2.0
    pools: dict[str, dict[tuple[str, ...], list[int]]] = {
        "replay": defaultdict(list),
        "kernel": defaultdict(list),
    }
    for index in replay:
        pools["replay"][_stratum(samples[index])].append(index)
    for index in kernel:
        pools["kernel"][_stratum(samples[index])].append(index)
    uses = {index: 0 for index in range(len(samples))}
    stratum_uses: defaultdict[tuple[str, ...], int] = defaultdict(int)
    chosen: list[int] = []
    replay_tokens = 0
    total_tokens = 0

    for _ in range(steps):
        candidates: list[tuple[float, float, str, int]] = []
        for domain in ("kernel", "replay"):
            strata = pools[domain]
            stratum = min(
                strata,
                key=lambda candidate: (
                    stratum_uses[candidate],
                    candidate,
                ),
            )
            pool = strata[stratum]
            index = min(
                pool,
                key=lambda candidate: (
                    (uses[candidate] + 1)
                    * _stable_rank(samples[candidate], candidate, config)[0],
                    _stable_rank(samples[candidate], candidate, config)[1],
                ),
            )
            tokens = samples[index].assistant_loss_tokens
            next_replay = replay_tokens + (tokens if domain == "replay" else 0)
            share = next_replay / (total_tokens + tokens)
            rank = _stable_rank(samples[index], index, config)[0]
            candidates.append((abs(share - midpoint), rank, domain, index))
        _, _, domain, index = min(candidates)
        chosen.append(index)
        uses[index] += 1
        stratum_uses[_stratum(samples[index])] += 1
        tokens = samples[index].assistant_loss_tokens
        total_tokens += tokens
        if domain == "replay":
            replay_tokens += tokens

    share = replay_tokens / total_tokens
    if not config.replay_min_share <= share <= config.replay_max_share:
        raise DataContractError(
            f"cannot meet replay loss-token gate in {steps} steps: "
            f"share={share:.6f}, required="
            f"[{config.replay_min_share:.2f}, {config.replay_max_share:.2f}]"
        )
    kernel_count = sum(not _is_replay(samples[index]) for index in chosen)
    replay_count = len(chosen) - kernel_count
    return SamplingPlan(
        indices=tuple(chosen),
        replay_loss_tokens=replay_tokens,
        total_loss_tokens=total_tokens,
        replay_share=share,
        sample_counts={"kernel": kernel_count, "general_coding": replay_count},
        loss_tokens={
            "kernel": total_tokens - replay_tokens,
            "general_coding": replay_tokens,
        },
    )
