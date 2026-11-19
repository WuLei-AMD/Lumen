"""Deterministic Qwen message formatting and assistant-only tokenization."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .contracts import DataContractError

SUPPORTED_MODEL_FAMILIES = frozenset(
    {"qwen", "qwen3-30b-a3b", "qwen/qwen3-30b-a3b"}
)
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert software engineer. Return only the requested patch."
)


class TokenizerLike(Protocol):
    def encode(self, text: str, **kwargs: Any) -> Sequence[int]: ...


class TokenizationOverflowError(DataContractError):
    """Raised when a sample exceeds the configured context window."""


@dataclass(frozen=True)
class TokenizationConfig:
    model_family: str = "qwen3-30b-a3b"
    max_length: int = 32768
    include_assistant_eot_in_loss: bool = True
    overflow_policy: str = "error"
    expected_chat_template_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.model_family.lower() not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(
                f"unsupported model_family: {self.model_family}; expected plain "
                "Qwen or Qwen3-30B-A3B"
            )
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.overflow_policy not in {"error", "quarantine"}:
            raise ValueError("overflow_policy must be error or quarantine")


@dataclass(frozen=True)
class TokenizedSample:
    sample_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    loss_mask: tuple[int, ...]
    assistant_loss_tokens: int
    sample_domain: str
    messages: tuple[Mapping[str, str], ...] = ()
    task_type: str = ""
    lane: str = ""
    implementation_family_id: str = ""
    primary_language: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "geak_qwen_messages_v1",
            "sample_id": self.sample_id,
            "messages": [dict(message) for message in self.messages],
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "loss_mask": list(self.loss_mask),
            "assistant_loss_tokens": self.assistant_loss_tokens,
            "sample_domain": self.sample_domain,
            "token_stats": {
                "assistant_loss_tokens": self.assistant_loss_tokens,
                "total_tokens": len(self.input_ids),
            },
            "source": {
                "task_type": self.task_type,
                "lane": self.lane,
                "implementation_family_id": self.implementation_family_id,
                "primary_language": self.primary_language,
            },
        }


@dataclass(frozen=True)
class QuarantinedSample:
    sample_id: str
    reason: str
    observed_length: int
    max_length: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "reason": self.reason,
            "observed_length": self.observed_length,
            "max_length": self.max_length,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def canonical_qwen_messages(
    sample: Mapping[str, Any],
    *,
    model_family: str = "qwen3-30b-a3b",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[dict[str, str], ...]:
    """Build stable system/user/assistant messages for plain Qwen/Qwen3."""

    family = model_family.lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"unsupported model_family: {model_family}")
    output = sample["output"]
    assistant = output.get("response", output.get("patch"))
    sample_input = sample["input"]
    if sample.get("schema_version") == "generic_coding_sft_v1":
        supplied = sample_input.get("messages")
        if supplied is not None:
            if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
                raise DataContractError("input.messages must be a message list")
            messages: list[dict[str, str]] = []
            for message in supplied:
                if not isinstance(message, Mapping):
                    raise DataContractError("input.messages entries must be objects")
                role, content = message.get("role"), message.get("content")
                if role not in {"system", "user"} or not isinstance(content, str):
                    raise DataContractError(
                        "generic input messages must contain only string system/user turns"
                    )
                messages.append({"role": role, "content": content})
            if not messages or messages[-1]["role"] != "user":
                raise DataContractError("generic input messages must end with a user turn")
            return (*messages, {"role": "assistant", "content": assistant})
        prompt = sample_input.get("prompt")
        if isinstance(prompt, str) and prompt:
            return (
                {
                    "role": "system",
                    "content": str(sample_input.get("system") or system_prompt),
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant},
            )
    user_payload = {
        "input": sample["input"],
        "task_type": sample["task_type"],
    }
    if sample.get("coding_task_type") is not None:
        user_payload["coding_task_type"] = sample["coding_task_type"]
    return (
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _canonical_json(user_payload)},
        {"role": "assistant", "content": assistant},
    )


def render_qwen_segments(
    messages: Sequence[Mapping[str, str]],
) -> tuple[str, str, str]:
    """Render canonical ChatML as prefix, assistant content, and assistant EOT."""

    roles = [message.get("role") for message in messages]
    if (
        not messages
        or roles[-1] != "assistant"
        or any(role not in {"system", "user"} for role in roles[:-1])
    ):
        raise DataContractError(
            "canonical Qwen messages must end with assistant after system/user turns"
        )
    prefix = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages[:-1]
    ) + "<|im_start|>assistant\n"
    return prefix, messages[-1]["content"], "<|im_end|>\n"


def _encode(tokenizer: TokenizerLike, text: str) -> tuple[int, ...]:
    try:
        values = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        values = tokenizer.encode(text)
    return tuple(int(value) for value in values)


def _normalize_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise DataContractError("tokenizer returned unsupported input_ids")
    return tuple(int(item) for item in value)


def _apply_chat_template(
    tokenizer: TokenizerLike,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> tuple[int, ...]:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        raise AttributeError("tokenizer has no apply_chat_template")
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        value = apply(list(messages), enable_thinking=False, **kwargs)
    except TypeError:
        value = apply(list(messages), **kwargs)
    return _normalize_ids(value)


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    return prefix


def _common_suffix(left: Sequence[int], right: Sequence[int], prefix: int) -> int:
    suffix = 0
    limit = min(len(left) - prefix, len(right) - prefix)
    while suffix < limit and left[-suffix - 1] == right[-suffix - 1]:
        suffix += 1
    return suffix


def _validate_chat_template_pin(
    tokenizer: TokenizerLike, expected_sha256: str | None
) -> None:
    if expected_sha256 is None:
        return
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        getter = getattr(tokenizer, "get_chat_template", None)
        template = getter() if callable(getter) else None
    if not isinstance(template, str) or not template:
        raise DataContractError("tokenizer has no chat template to verify")
    actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if actual != expected_sha256:
        raise DataContractError(
            f"chat template SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )


def tokenize_sample(
    sample: Mapping[str, Any],
    tokenizer: TokenizerLike,
    config: TokenizationConfig,
) -> TokenizedSample | QuarantinedSample:
    """Pre-tokenize one sample without truncation.

    Segments are encoded independently so the mask is explicit and does not
    depend on tokenizer-specific offset mappings. For Qwen, ChatML boundaries
    are special-token boundaries, making this equivalent to the canonical
    template while remaining easy to audit.
    """

    messages = canonical_qwen_messages(sample, model_family=config.model_family)
    _validate_chat_template_pin(tokenizer, config.expected_chat_template_sha256)
    try:
        input_ids = _apply_chat_template(
            tokenizer, messages, add_generation_prompt=False
        )
        prompt_ids = _apply_chat_template(
            tokenizer, messages[:-1], add_generation_prompt=True
        )
        empty_ids = _apply_chat_template(
            tokenizer,
            (*messages[:-1], {"role": "assistant", "content": ""}),
            add_generation_prompt=False,
        )
        prefix_length = _common_prefix(input_ids, empty_ids)
        suffix_length = _common_suffix(input_ids, empty_ids, prefix_length)
        content_end = len(input_ids) - suffix_length
        if content_end <= prefix_length:
            raise DataContractError(
                f"{sample['sample_id']}: assistant produced no distinct tokens"
            )
        if tuple(input_ids[: len(prompt_ids)]) == prompt_ids:
            prefix_length = len(prompt_ids)
        loss_end = len(input_ids) if config.include_assistant_eot_in_loss else content_end
        loss_mask = (
            (0,) * prefix_length
            + (1,) * (loss_end - prefix_length)
            + (0,) * (len(input_ids) - loss_end)
        )
    except AttributeError:
        prefix, assistant, eot = render_qwen_segments(messages)
        prefix_ids = _encode(tokenizer, prefix)
        assistant_ids = _encode(tokenizer, assistant)
        eot_ids = _encode(tokenizer, eot)
        input_ids = prefix_ids + assistant_ids + eot_ids
        eot_mask = 1 if config.include_assistant_eot_in_loss else 0
        loss_mask = (
            (0,) * len(prefix_ids)
            + (1,) * len(assistant_ids)
            + (eot_mask,) * len(eot_ids)
        )
    if len(input_ids) > config.max_length:
        overflow = QuarantinedSample(
            sample_id=str(sample["sample_id"]),
            reason="context_overflow",
            observed_length=len(input_ids),
            max_length=config.max_length,
        )
        if config.overflow_policy == "quarantine":
            return overflow
        raise TokenizationOverflowError(
            f"{sample['sample_id']}: tokenized length {len(input_ids)} exceeds "
            f"{config.max_length}; truncation is forbidden"
        )
    if sum(loss_mask) == 0:
        raise DataContractError(f"{sample['sample_id']}: assistant has no tokens")
    provenance = sample.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    contract = sample.get("input", {}).get("contract", {})
    if not isinstance(contract, Mapping):
        contract = {}
    return TokenizedSample(
        sample_id=str(sample["sample_id"]),
        input_ids=input_ids,
        attention_mask=(1,) * len(input_ids),
        loss_mask=loss_mask,
        assistant_loss_tokens=sum(loss_mask),
        sample_domain=str(sample.get("sample_domain", "kernel")),
        messages=tuple(messages),
        task_type=str(sample.get("task_type", "")),
        lane=str(provenance.get("lane", "")),
        implementation_family_id=str(
            provenance.get("implementation_family_id", "")
        ),
        primary_language=str(
            sample.get("primary_language")
            or provenance.get("language")
            or contract.get("language")
            or ""
        ),
    )


def tokenize_samples(
    samples: Sequence[Mapping[str, Any]],
    tokenizer: TokenizerLike,
    config: TokenizationConfig,
) -> tuple[list[TokenizedSample], list[QuarantinedSample]]:
    """Tokenize a collection and return explicit overflow quarantine records."""

    accepted: list[TokenizedSample] = []
    quarantined: list[QuarantinedSample] = []
    for sample in samples:
        result = tokenize_sample(sample, tokenizer, config)
        if isinstance(result, QuarantinedSample):
            quarantined.append(result)
        else:
            accepted.append(result)
    return accepted, quarantined
