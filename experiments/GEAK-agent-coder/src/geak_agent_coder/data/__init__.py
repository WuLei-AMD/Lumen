"""Fail-closed data contract for GEAK agent-coder SFT."""

from .admission import admit_row, path_sha256, verify_source_admission
from .build import build_data, build_from_config, load_local_tokenizer
from .config import load_data_config
from .contracts import (
    DataContractError,
    ReplayGateError,
    ReplayGateResult,
    SchemaValidationError,
    TrainingSplitError,
    enforce_final_replay_gates,
    iter_local_samples,
    load_samples,
    load_training_samples,
    validate_sample,
)
from .formatting import (
    QuarantinedSample,
    TokenizationConfig,
    TokenizationOverflowError,
    TokenizedSample,
    canonical_qwen_messages,
    render_qwen_segments,
    tokenize_sample,
    tokenize_samples,
)
from .manifest import (
    aggregate_stats,
    build_data_manifest,
    write_manifest,
    write_tokenized_jsonl,
)
from .mapping import evaluate_mapping, map_row, register_adapter
from .sampling import SamplingConfig, SamplingPlan, build_sampling_plan
from .sources import iter_source, register_source

__all__ = [
    "DataContractError",
    "QuarantinedSample",
    "ReplayGateError",
    "ReplayGateResult",
    "SamplingConfig",
    "SamplingPlan",
    "SchemaValidationError",
    "TokenizationConfig",
    "TokenizationOverflowError",
    "TokenizedSample",
    "TrainingSplitError",
    "admit_row",
    "aggregate_stats",
    "build_data",
    "build_data_manifest",
    "build_from_config",
    "build_sampling_plan",
    "canonical_qwen_messages",
    "enforce_final_replay_gates",
    "evaluate_mapping",
    "iter_local_samples",
    "iter_source",
    "load_data_config",
    "load_local_tokenizer",
    "load_samples",
    "load_training_samples",
    "map_row",
    "path_sha256",
    "register_adapter",
    "register_source",
    "render_qwen_segments",
    "tokenize_sample",
    "tokenize_samples",
    "validate_sample",
    "verify_source_admission",
    "write_manifest",
    "write_tokenized_jsonl",
]
