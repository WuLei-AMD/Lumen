from __future__ import annotations

import hashlib

import pytest

from geak_agent_coder.data import (
    DataContractError,
    QuarantinedSample,
    TokenizationConfig,
    TokenizationOverflowError,
    canonical_qwen_messages,
    render_qwen_segments,
    tokenize_sample,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


class FakeChatTokenizer(FakeTokenizer):
    chat_template = "fake-qwen-template-v1"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ):
        assert tokenize
        assert not enable_thinking
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return self.encode(rendered)


def sample() -> dict:
    return {
        "schema_version": "general_coding_replay_v1",
        "sample_id": "gc-1",
        "sample_domain": "general_coding",
        "task_type": "general_coding_replay",
        "coding_task_type": "bug_fix",
        "split": "train",
        "input": {"z": 1, "a": "two"},
        "output": {"patch": "PATCH"},
        "provenance": {"dataset": "local"},
    }


def test_canonical_messages_are_sorted_and_support_model_id() -> None:
    messages = canonical_qwen_messages(
        sample(), model_family="Qwen/Qwen3-30B-A3B"
    )
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[1]["content"] == (
        '{"coding_task_type":"bug_fix","input":{"a":"two","z":1},'
        '"task_type":"general_coding_replay"}'
    )
    assert messages[2]["content"] == "PATCH"
    assert messages == canonical_qwen_messages(
        sample(), model_family="Qwen/Qwen3-30B-A3B"
    )


@pytest.mark.parametrize("include_eot", [True, False])
def test_stored_masks_cover_assistant_content_and_configured_eot(
    include_eot: bool,
) -> None:
    tokenizer = FakeTokenizer()
    config = TokenizationConfig(
        model_family="Qwen/Qwen3-30B-A3B",
        max_length=1000,
        include_assistant_eot_in_loss=include_eot,
    )
    result = tokenize_sample(sample(), tokenizer, config)
    assert not isinstance(result, QuarantinedSample)
    messages = canonical_qwen_messages(sample())
    prefix, assistant, eot = render_qwen_segments(messages)
    assert len(result.input_ids) == len(prefix + assistant + eot)
    assert result.attention_mask == (1,) * len(result.input_ids)
    assert result.loss_mask[: len(prefix)] == (0,) * len(prefix)
    assert result.loss_mask[
        len(prefix) : len(prefix) + len(assistant)
    ] == (1,) * len(assistant)
    assert result.loss_mask[-len(eot) :] == (
        (1 if include_eot else 0),
    ) * len(eot)
    assert result.assistant_loss_tokens == len(assistant) + (
        len(eot) if include_eot else 0
    )
    stored = result.as_dict()
    assert stored["input_ids"] == list(result.input_ids)
    assert stored["loss_mask"] == list(result.loss_mask)


def test_overflow_never_silently_truncates() -> None:
    tokenizer = FakeTokenizer()
    with pytest.raises(TokenizationOverflowError, match="truncation is forbidden"):
        tokenize_sample(
            sample(),
            tokenizer,
            TokenizationConfig(max_length=10, overflow_policy="error"),
        )

    result = tokenize_sample(
        sample(),
        tokenizer,
        TokenizationConfig(max_length=10, overflow_policy="quarantine"),
    )
    assert isinstance(result, QuarantinedSample)
    assert result.reason == "context_overflow"
    assert result.observed_length > result.max_length


def test_rejects_unknown_chat_family() -> None:
    with pytest.raises(ValueError, match="unsupported model_family"):
        TokenizationConfig(model_family="llama")


def test_uses_pinned_chat_template_and_masks_content_plus_eot() -> None:
    tokenizer = FakeChatTokenizer()
    expected = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    result = tokenize_sample(
        sample(),
        tokenizer,
        TokenizationConfig(
            max_length=1000,
            expected_chat_template_sha256=expected,
            include_assistant_eot_in_loss=True,
        ),
    )
    assert not isinstance(result, QuarantinedSample)
    prompt = tokenizer.apply_chat_template(
        canonical_qwen_messages(sample())[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert result.loss_mask[: len(prompt)] == (0,) * len(prompt)
    assert all(result.loss_mask[len(prompt) :])
    assert result.as_dict()["schema_version"] == "geak_qwen_messages_v1"
    assert result.as_dict()["messages"][-1]["content"] == "PATCH"

    with pytest.raises(DataContractError, match="chat template SHA256 mismatch"):
        tokenize_sample(
            sample(),
            tokenizer,
            TokenizationConfig(
                max_length=1000,
                expected_chat_template_sha256="0" * 64,
            ),
        )
