"""Safe declarative mapping of source rows into canonical records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from .contracts import DataContractError

Adapter = Callable[[Any], Any]
_ADAPTERS: dict[str, Adapter] = {}


def register_adapter(name: str, adapter: Adapter) -> None:
    """Register an explicitly supplied adapter; import strings are never resolved."""

    if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
        raise ValueError("adapter name must be a simple identifier")
    if not callable(adapter):
        raise TypeError("adapter must be callable")
    if name in _ADAPTERS:
        raise ValueError(f"adapter already registered: {name}")
    _ADAPTERS[name] = adapter


def _path(row: Mapping[str, Any], path: str) -> Any:
    current: Any = row
    if not path:
        raise DataContractError("source path must not be empty")
    for component in path.split("."):
        if component in {"", "__class__", "__dict__", "__globals__"}:
            raise DataContractError(f"unsafe source path: {path}")
        if isinstance(current, Mapping) and component in current:
            current = current[component]
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ) and component.isdigit() and int(component) < len(current):
            current = current[int(component)]
        else:
            raise DataContractError(f"missing source field: {path}")
    return deepcopy(current)


def _pick(row: Mapping[str, Any], choices: Any) -> Any:
    if not isinstance(choices, list) or not choices:
        raise DataContractError("pick/fallback requires a non-empty list")
    errors = []
    for choice in choices:
        try:
            value = evaluate_mapping(choice, row)
        except DataContractError as exc:
            errors.append(str(exc))
            continue
        if value is not None:
            return value
    raise DataContractError("no pick/fallback value resolved: " + "; ".join(errors))


def evaluate_mapping(spec: Any, row: Mapping[str, Any]) -> Any:
    """Evaluate the closed mapping DSL against one raw row."""

    if not isinstance(spec, Mapping):
        raise DataContractError("mapping expressions must be objects")
    operators = set(spec) & {
        "source",
        "const",
        "pick",
        "fallback",
        "object",
        "wrap",
        "adapter",
    }
    if len(operators) != 1:
        raise DataContractError("mapping expression must contain exactly one operator")
    operator = next(iter(operators))
    extra = set(spec) - {operator}
    if operator == "source":
        if extra:
            raise DataContractError(f"unexpected source options: {sorted(extra)}")
        path = spec["source"]
        if not isinstance(path, str):
            raise DataContractError("source path must be a string")
        return _path(row, path)
    if operator == "const":
        if extra:
            raise DataContractError(f"unexpected const options: {sorted(extra)}")
        return deepcopy(spec["const"])
    if operator in {"pick", "fallback"}:
        if extra:
            raise DataContractError(f"unexpected {operator} options: {sorted(extra)}")
        return _pick(row, spec[operator])
    if operator == "object":
        if extra:
            raise DataContractError(f"unexpected object options: {sorted(extra)}")
        fields = spec["object"]
        if not isinstance(fields, Mapping):
            raise DataContractError("object mapping must be a mapping")
        return {str(key): evaluate_mapping(value, row) for key, value in fields.items()}
    if operator == "wrap":
        if extra:
            raise DataContractError(f"unexpected wrap options: {sorted(extra)}")
        wrapper = spec["wrap"]
        if not isinstance(wrapper, Mapping) or set(wrapper) != {"key", "value"}:
            raise DataContractError("wrap requires exactly key and value")
        key = wrapper["key"]
        if not isinstance(key, str) or not key:
            raise DataContractError("wrap key must be a non-empty string")
        return {key: evaluate_mapping(wrapper["value"], row)}
    if extra != {"value"}:
        raise DataContractError("adapter requires exactly adapter and value")
    name = spec["adapter"]
    if not isinstance(name, str) or name not in _ADAPTERS:
        raise DataContractError(f"unknown adapter: {name!r}")
    return _ADAPTERS[name](evaluate_mapping(spec["value"], row))


def map_row(row: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise DataContractError("source row must be an object")
    if not isinstance(mapping, Mapping):
        raise DataContractError("field_mapping must be an object")
    return {
        str(field): evaluate_mapping(expression, row)
        for field, expression in mapping.items()
    }


register_adapter("identity", lambda value: value)
register_adapter("strip", lambda value: value.strip() if isinstance(value, str) else value)
register_adapter("string", lambda value: str(value))
