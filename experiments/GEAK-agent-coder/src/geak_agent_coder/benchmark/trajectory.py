"""Parse MultiTune trajectory and ToolAgentLoop events into per-turn records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .models import TurnRecord


class TrajectoryFormatError(ValueError):
    pass


def parse_trajectory(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    task_id: str | None = None,
    variant: str | None = None,
    seed: int | None = None,
    operator: str = "unknown",
    lane: str = "unknown",
) -> list[TurnRecord]:
    """Parse direct turn rows, trajectory.jsonl wrappers, or agent-loop events."""

    state = {
        "task_id": task_id,
        "variant": variant,
        "seed": seed,
        "operator": operator,
        "lane": lane,
        "tokens_input": 0,
        "tokens_output": 0,
        "wall_time_s": 0.0,
        "tool_calls": 0,
        "turn": 0,
        "turn_emitted": False,
    }
    records: list[TurnRecord] = []
    for raw in _iter_rows(source):
        if _looks_like_turn(raw):
            records.append(_direct_turn(raw, state))
            continue
        for event in _flatten_event(raw):
            _update_identity(state, event)
            event_type = str(event.get("type") or event.get("event") or "")
            payload = event.get("payload")
            payload = payload if isinstance(payload, Mapping) else event
            if event_type in {"model", "model_turn", "assistant_turn"}:
                _flush_unevaluated(records, state)
                _reset_turn_cost(state)
                state["turn"] += 1
                state["turn_emitted"] = False
                usage = payload.get("usage")
                usage = usage if isinstance(usage, Mapping) else {}
                state["tokens_input"] += _integer(
                    usage, "prompt_tokens", "input_tokens", default=0
                )
                state["tokens_output"] += _integer(
                    usage, "completion_tokens", "output_tokens", default=0
                )
                state["wall_time_s"] += _number(
                    payload, "elapsed_seconds", "wall_time_s", default=0.0
                )
                continue
            if event_type in {"tool", "tool_result", "evaluation", "evaluate"}:
                state["tool_calls"] += 1
                state["wall_time_s"] += _number(
                    payload, "elapsed_seconds", "wall_time_s", default=0.0
                )
                evaluation = _find_evaluation(payload)
                if evaluation is not None:
                    if state["turn"] == 0:
                        state["turn"] = 1
                    record = _evaluation_turn(evaluation, payload, state)
                    if state["turn_emitted"] and records:
                        records[-1] = record
                    else:
                        records.append(record)
                    state["turn_emitted"] = True
    _flush_unevaluated(records, state)
    records.sort(key=lambda item: (item.task_id, item.variant, item.seed, item.turn))
    return records


def write_turn_jsonl(records: Iterable[TurnRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def _iter_rows(
    source: str | Path | Iterable[Mapping[str, Any]],
) -> Iterator[Mapping[str, Any]]:
    if not isinstance(source, (str, Path)):
        yield from source
        return
    path = Path(source)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrajectoryFormatError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, Mapping):
                raise TrajectoryFormatError(f"{path}:{line_number}: expected object")
            yield value


def _flatten_event(row: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    payload = row.get("payload")
    if str(row.get("event") or "") == "agent_loop" and isinstance(payload, Mapping):
        events = payload.get("events")
        if isinstance(events, list):
            for event in events:
                if isinstance(event, Mapping):
                    yield {
                        key: payload[key]
                        for key in ("task_id", "variant", "seed", "operator", "lane")
                        if key in payload
                    } | dict(event)
            return
    events = row.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, Mapping):
                yield event
        return
    yield row


def _find_evaluation(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates: list[Any] = [payload.get("evaluation")]
    result = payload.get("result")
    if isinstance(result, Mapping):
        candidates.extend([result.get("evaluation"), result])
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        candidates.append(metrics.get("evaluation"))
    for value in candidates:
        if isinstance(value, Mapping) and (
            "compiled" in value or "correct" in value or "speedup_geomean" in value
        ):
            return value
    return None


def _evaluation_turn(
    evaluation: Mapping[str, Any],
    payload: Mapping[str, Any],
    state: Mapping[str, Any],
) -> TurnRecord:
    compiled = bool(evaluation.get("compiled", False))
    correct = bool(evaluation.get("correct", False))
    speedup = evaluation.get("speedup_vs_frozen_baseline")
    if speedup is None:
        speedup = evaluation.get("speedup_geomean")
    return TurnRecord(
        task_id=_identity(state, "task_id"),
        variant=_identity(state, "variant"),
        seed=int(state["seed"] if state["seed"] is not None else 0),
        turn=int(state["turn"]),
        compiled=compiled,
        correct=correct,
        evaluated=True,
        speedup_vs_frozen_baseline=(
            float(speedup) if isinstance(speedup, (int, float)) else None
        ),
        tokens_input=int(state["tokens_input"]),
        tokens_output=int(state["tokens_output"]),
        wall_time_s=float(state["wall_time_s"]),
        error_type=_error_type(evaluation, payload, compiled, correct),
        operator=str(state["operator"]),
        lane=str(state["lane"]),
        tool_calls=int(state["tool_calls"]),
    )


def _direct_turn(row: Mapping[str, Any], defaults: Mapping[str, Any]) -> TurnRecord:
    merged = dict(defaults)
    merged.update(row)
    speedup = merged.get("speedup_vs_frozen_baseline")
    return TurnRecord(
        task_id=_identity(merged, "task_id"),
        variant=_identity(merged, "variant"),
        seed=int(merged.get("seed") or 0),
        turn=int(merged["turn"]),
        compiled=bool(merged.get("compiled")),
        correct=bool(merged.get("correct")),
        evaluated=bool(merged.get("evaluated", True)),
        speedup_vs_frozen_baseline=(
            float(speedup) if isinstance(speedup, (int, float)) else None
        ),
        tokens_input=int(merged.get("tokens_input") or 0),
        tokens_output=int(merged.get("tokens_output") or 0),
        wall_time_s=float(merged.get("wall_time_s") or 0.0),
        error_type=(
            str(merged["error_type"]) if merged.get("error_type") is not None else None
        ),
        operator=str(merged.get("operator") or "unknown"),
        lane=str(merged.get("lane") or "unknown"),
        tool_calls=int(merged.get("tool_calls") or 0),
        metadata=(
            dict(merged["metadata"])
            if isinstance(merged.get("metadata"), Mapping)
            else {}
        ),
    )


def _update_identity(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    payload = event.get("payload")
    sources = [event, payload] if isinstance(payload, Mapping) else [event]
    for source in sources:
        for key in ("task_id", "variant", "seed", "operator", "lane"):
            if source.get(key) is not None and state.get(key) in (None, "unknown"):
                state[key] = source[key]


def _reset_turn_cost(state: dict[str, Any]) -> None:
    state.update(
        tokens_input=0,
        tokens_output=0,
        wall_time_s=0.0,
        tool_calls=0,
    )


def _flush_unevaluated(
    records: list[TurnRecord], state: dict[str, Any]
) -> None:
    if int(state["turn"]) <= 0 or bool(state["turn_emitted"]):
        return
    records.append(
        TurnRecord(
            task_id=_identity(state, "task_id"),
            variant=_identity(state, "variant"),
            seed=int(state["seed"] if state["seed"] is not None else 0),
            turn=int(state["turn"]),
            compiled=False,
            correct=False,
            evaluated=False,
            tokens_input=int(state["tokens_input"]),
            tokens_output=int(state["tokens_output"]),
            wall_time_s=float(state["wall_time_s"]),
            error_type=None,
            operator=str(state["operator"]),
            lane=str(state["lane"]),
            tool_calls=int(state["tool_calls"]),
        )
    )
    state["turn_emitted"] = True


def _error_type(
    evaluation: Mapping[str, Any],
    payload: Mapping[str, Any],
    compiled: bool,
    correct: bool,
) -> str | None:
    explicit = evaluation.get("error_type") or payload.get("error_type")
    if explicit:
        return str(explicit)
    if not compiled:
        return "compile"
    if not correct:
        return "correctness"
    return None


def _looks_like_turn(row: Mapping[str, Any]) -> bool:
    return "turn" in row and "compiled" in row and "correct" in row


def _identity(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if item is None or not str(item).strip():
        raise TrajectoryFormatError(f"missing {key}; pass it explicitly or record it")
    return str(item)


def _number(
    value: Mapping[str, Any], *keys: str, default: float
) -> float:
    for key in keys:
        item = value.get(key)
        if isinstance(item, (int, float)):
            return float(item)
    return default


def _integer(
    value: Mapping[str, Any], *keys: str, default: int
) -> int:
    return int(_number(value, *keys, default=float(default)))
