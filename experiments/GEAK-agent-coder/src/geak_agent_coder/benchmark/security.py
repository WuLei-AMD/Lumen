"""Fail-closed checks for model-visible held-out task material."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping


class HeldOutLeakageError(ValueError):
    """Raised before a held-out task can reach a model endpoint."""


_FORBIDDEN_KEY = re.compile(
    r"(?:"
    r"(?:^|_)target$|"
    r"target_(?:patch|source|output|solution)|"
    r"reference_(?:patch|source|output|solution|path)|"
    r"(?:^|_)(?:solution|answer|gold|oracle)(?:_|$)|"
    r"expected_(?:output|patch|answer)|"
    r"ground_truth"
    r")",
    re.IGNORECASE,
)
_ALLOWED_ID_KEY = re.compile(r"(?:^|_)(?:id|hash|digest)$", re.IGNORECASE)


def assert_no_held_out_leakage(
    model_visible: Mapping[str, Any],
    *,
    protected_sha256: Iterable[str],
    held_out: bool = True,
) -> None:
    """Reject target-like fields and exact protected-content fingerprints.

    A held-out manifest must provide at least one protected digest. This makes a
    missing protection catalog a hard error rather than silently weakening the
    benchmark.
    """

    if not held_out:
        return
    digests = {str(item).lower() for item in protected_sha256 if str(item).strip()}
    if not digests:
        raise HeldOutLeakageError(
            "held-out execution requires non-empty protected_sha256 fingerprints"
        )
    for path, value in _walk(model_visible):
        key = path[-1] if path else ""
        if _FORBIDDEN_KEY.search(key) and not _ALLOWED_ID_KEY.search(key):
            raise HeldOutLeakageError(
                f"model-visible held-out field is target-like: {'.'.join(path)}"
            )
        if isinstance(value, str):
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            if digest in digests:
                raise HeldOutLeakageError(
                    f"model-visible value matches protected content: {'.'.join(path)}"
                )
            # Structured strings must not bypass the key walk.
            stripped = value.lstrip()
            if stripped.startswith(("{", "[")):
                try:
                    nested = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(nested, (dict, list)):
                    assert_no_held_out_leakage(
                        {"embedded": nested},
                        protected_sha256=digests,
                        held_out=True,
                    )


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = path + (str(key),)
            yield item_path, item
            yield from _walk(item, item_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            item_path = path + (str(index),)
            yield item_path, item
            yield from _walk(item, item_path)
