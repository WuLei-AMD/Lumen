"""Import-light value objects shared by benchmark parsing and metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TurnRecord:
    task_id: str
    variant: str
    seed: int
    turn: int
    compiled: bool
    correct: bool
    evaluated: bool = True
    speedup_vs_frozen_baseline: float | None = None
    tokens_input: int = 0
    tokens_output: int = 0
    wall_time_s: float = 0.0
    error_type: str | None = None
    operator: str = "unknown"
    lane: str = "unknown"
    tool_calls: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tokens_total(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def passed(self) -> bool:
        return self.compiled and self.correct

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tokens_total"] = self.tokens_total
        return value


@dataclass(frozen=True)
class FixedBenchmarkIdentity:
    agent_hash: str
    suite_hash: str
    decode_hash: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FixedBenchmarkIdentity":
        return cls(
            agent_hash=_required_text(value, "agent_hash"),
            suite_hash=_required_text(value, "suite_hash"),
            decode_hash=_required_text(value, "decode_hash"),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ModelVariant:
    name: str
    base_url: str
    model: str
    checkpoint_identity: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelVariant":
        return cls(
            name=_required_text(value, "name"),
            base_url=_required_text(value, "base_url"),
            model=_required_text(value, "model"),
            checkpoint_identity=_required_text(value, "checkpoint_identity"),
        )

    def public_dict(self) -> dict[str, str]:
        return asdict(self)


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item.strip()
