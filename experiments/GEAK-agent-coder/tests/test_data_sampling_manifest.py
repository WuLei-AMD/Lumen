from __future__ import annotations

import json
from pathlib import Path

import pytest

from geak_agent_coder.data import (
    DataContractError,
    QuarantinedSample,
    SamplingConfig,
    TokenizedSample,
    build_data_manifest,
    build_sampling_plan,
    write_manifest,
    write_tokenized_jsonl,
)


def tokenized(
    sample_id: str,
    domain: str,
    *,
    sequence_tokens: int,
    loss_tokens: int,
    lane: str = "",
    task_type: str = "",
    family: str = "",
    language: str = "",
) -> TokenizedSample:
    return TokenizedSample(
        sample_id=sample_id,
        input_ids=(1,) * sequence_tokens,
        attention_mask=(1,) * sequence_tokens,
        loss_mask=(0,) * (sequence_tokens - loss_tokens)
        + (1,) * loss_tokens,
        assistant_loss_tokens=loss_tokens,
        sample_domain=domain,
        lane=lane,
        task_type=task_type,
        implementation_family_id=family,
        primary_language=language,
    )


def source(sample_id: str, domain: str) -> dict:
    schema = (
        "general_coding_replay_v1"
        if domain == "general_coding"
        else "geak_kernel_sft_v1"
    )
    return {
        "schema_version": schema,
        "sample_id": sample_id,
        "split": "train",
        "sample_domain": domain,
        "task_type": (
            "general_coding_replay"
            if domain == "general_coding"
            else "cold_start"
        ),
    }


def test_sampler_is_deterministic_bounded_and_length_aware() -> None:
    samples = [
        tokenized("k-short", "kernel", sequence_tokens=20, loss_tokens=10),
        tokenized("k-long", "kernel", sequence_tokens=200, loss_tokens=10),
        tokenized("r-short", "general_coding", sequence_tokens=20, loss_tokens=10),
        tokenized("r-long", "general_coding", sequence_tokens=200, loss_tokens=10),
    ]
    config = SamplingConfig(seed=17)
    first = build_sampling_plan(samples, steps=200, config=config)
    second = build_sampling_plan(samples, steps=200, config=config)

    assert first == second
    assert 0.15 <= first.replay_share <= 0.20
    counts = {index: first.indices.count(index) for index in range(len(samples))}
    assert counts[0] > counts[1]
    assert counts[2] > counts[3]
    assert first.total_loss_tokens == sum(
        samples[index].assistant_loss_tokens for index in first.indices
    )


def test_sampler_reports_impossible_small_schedule() -> None:
    samples = [
        tokenized("k", "kernel", sequence_tokens=10, loss_tokens=10),
        tokenized("r", "general_coding", sequence_tokens=10, loss_tokens=10),
    ]
    with pytest.raises(DataContractError, match="cannot meet replay"):
        build_sampling_plan(samples, steps=2)


def test_sampler_balances_kernel_and_replay_strata() -> None:
    samples = [
        tokenized(
            "k-hip",
            "kernel",
            sequence_tokens=20,
            loss_tokens=10,
            lane="hip_gfx942",
            task_type="cold_start",
            family="hip-a",
        ),
        tokenized(
            "k-triton",
            "kernel",
            sequence_tokens=20,
            loss_tokens=10,
            lane="triton_gfx942",
            task_type="error_recovery",
            family="triton-a",
        ),
        tokenized(
            "r-python",
            "general_coding",
            sequence_tokens=20,
            loss_tokens=10,
            task_type="general_coding_replay",
            language="python",
        ),
        tokenized(
            "r-rust",
            "general_coding",
            sequence_tokens=20,
            loss_tokens=10,
            task_type="general_coding_replay",
            language="rust",
        ),
    ]
    plan = build_sampling_plan(samples, steps=200)
    counts = [plan.indices.count(index) for index in range(len(samples))]
    assert abs(counts[0] - counts[1]) <= 1
    assert abs(counts[2] - counts[3]) <= 1


def test_manifest_contains_aggregate_stats_quarantine_and_sampling(
    tmp_path: Path,
) -> None:
    token_rows = [
        tokenized("k", "kernel", sequence_tokens=20, loss_tokens=10),
        tokenized("r", "general_coding", sequence_tokens=20, loss_tokens=10),
    ]
    source_rows = [source("k", "kernel"), source("r", "general_coding")]
    plan = build_sampling_plan(token_rows, steps=100)
    quarantine = [QuarantinedSample("too-long", "context_overflow", 99, 32)]
    manifest = build_data_manifest(
        source_rows,
        token_rows,
        tokenization_config={
            "include_assistant_eot_in_loss": True,
            "max_length": 32768,
        },
        quarantined=quarantine,
        sampling_plan=plan,
    )
    assert manifest["schema_version"] == "geak_training_data_manifest_v1"
    assert manifest["stats"]["domain_counts"] == {
        "general_coding": 1,
        "kernel": 1,
    }
    assert manifest["stats"]["assistant_loss_tokens"] == {
        "general_coding": 10,
        "kernel": 10,
    }
    assert manifest["quarantine"][0]["sample_id"] == "too-long"
    assert 0.15 <= manifest["sampling"]["replay_share"] <= 0.20

    output = tmp_path / "manifest.json"
    write_manifest(output, manifest)
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert output.read_text(encoding="utf-8").endswith("\n")

    tokenized_output = tmp_path / "tokenized.jsonl"
    write_tokenized_jsonl(tokenized_output, token_rows)
    stored = [
        json.loads(line)
        for line in tokenized_output.read_text(encoding="utf-8").splitlines()
    ]
    assert stored[0]["input_ids"] == [1] * 20
    assert stored[0]["attention_mask"] == [1] * 20
    assert stored[0]["loss_mask"] == [0] * 10 + [1] * 10
