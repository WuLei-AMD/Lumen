# Phase 1 Data Contract

## Inputs

The training export combines:

- Kernel Train: `geak_kernel_sft_v1`, `split=train`;
- Kernel Dev: the same schema with `split=dev`, evaluation only;
- General Coding Replay: `general_coding_replay_v1`, exactly 500 accepted
  training rows after finalization.

Held-out kernel and coding tasks are never loaded by the exporter or trainer.

## Required replay gates

General Coding Replay is admitted only when all authoritative reports agree:

```text
finalization.accepted == 500
finalization.leakage_status == "pass"
finalization.package_status in {"ready", "pass"}
leakage.status == "pass"
package_gate.status in {"ready", "pass"}
trust.offline_cpu_only_double_replay_passed == 500
```

Every referenced input and manifest checksum is pinned into the export
manifest. Missing reports, unknown status values, duplicate IDs, Dev/held-out
rows, or checksum mismatches fail closed.

## Qwen record

The exporter writes `geak_qwen_messages_v1` records containing:

- canonical system/user/assistant messages;
- tokenizer-produced `input_ids` and `attention_mask`;
- an explicit `loss_mask`;
- split/domain/task/language/lineage metadata;
- prompt, assistant, loss and total token counts.

The loss mask covers assistant content and, by the Phase 1 model contract, the
assistant end-of-turn suffix. The assistant role header, prompt, harness
output and validation labels are excluded.

No record is silently truncated. Overlength rows either stop export or are
written to an explicit quarantine with their full token statistics.

## Sampling

Kernel rows are balanced by lane, task type and implementation family.
General coding rows are mixed by assistant loss tokens, not row count:

```text
general_loss_tokens / total_loss_tokens ∈ [0.15, 0.20]
```

Sampling is deterministic for a pinned seed and export manifest. Dev is kept
in a separate loader and never receives gradients.

## Config-driven raw sources

`geak_data_config_v2` accepts `local_jsonl`, `local_hf_layout`, and an
offline-by-default `hf_cache` source. A closed mapping DSL supports nested
field reads, constants, fallback selection, object/wrap construction, and a
small registered adapter set. Arbitrary import strings are not executed.

Mappings must produce one of:

- `geak_kernel_sft_v1` for independently verified kernel patches;
- `general_coding_replay_v1` for the strict GEAK replay release only;
- `generic_coding_sft_v1` for other coding prompts/conversations and assistant
  responses.

Every mapped row passes canonical schema, split and provenance validation.
Declaring a replay row under a generic policy is rejected. Generic datasets do
not inherit the GEAK-specific 500-row requirement, but their source revision,
checksums and generated manifest must still be pinned.

`data-build` materializes the sampling plan into `train.tokenized.jsonl`
instead of merely recording planned indices. Training authorization verifies
the manifest SHA256, tokenizer identity and emitted training-file SHA256.

## SFT-f Messages Format Dataset (2026-10-11)

### New data source: unified messages format

SFT-f bypasses the `geak_kernel_sft_v1` exporter and uses a direct messages
chat format tokenizer (`scripts/tokenize_messages_dataset.py`). This enables
training on both patch optimization and complete kernel implementation data.

### Input format

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert GPU kernel engineer..."},
    {"role": "user", "content": "<task description + kernel code>"},
    {"role": "assistant", "content": "<diff patch or complete kernel>"}
  ],
  "task_type": "direction_conditioned|complete_implementation|...",
  "operator": "gemm|rms_norm|...",
  "schema_version": "messages_v1"
}
```

### Dataset composition (train.jsonl: 3164 samples)

| Task Type | Count | Input | Output |
|-----------|-------|-------|--------|
| direction_conditioned | 907 | kernel + optimization direction | diff patch |
| complete_implementation | 1164 | operator spec (no code) | complete kernel |
| cold_start | 307 | kernel, no direction | diff patch |
| error_recovery | 293 | broken kernel + error | fix patch |
| profile_guided | 293 | kernel + profiling data | optimization patch |
| regression_balance | 200 | kernel + constraints | balanced patch |

### Tokenization

Uses `tokenizer.apply_chat_template()` directly on the messages list.
Loss mask: `0` for system/user prompt tokens, `1` for assistant content + EOT.

```text
train: 3164 samples, 6.1M total tokens, 3.5M loss tokens, avg 1937 tok/sample
dev:   200 samples, 821K total tokens, 142K loss tokens, avg 4106 tok/sample
```

### Data sources

- **patch_v1** (2000): `phase1-production-wave-2000-v1` verified agent runs
- **complete_impl_v2** (1164): Fuzzy patch apply on wave-2000 parent→optimized
- **dev** (200): `phase1-dev-wave-200-v1` (evaluation only)

### No overlap with benchmark

Training data from `phase1-production-wave-*` (23 operators).
Benchmark from `held-out-benchmark-aiter` (8 operators, 100 tasks).
Verified: 0 content overlap.

### Build pipeline

```bash
# 1. Generate complete implementation samples
python3 scripts/gen_complete_impl.py

# 2. Unify to messages format + upload to HF
python3 scripts/unify_sft_format.py

# 3. Tokenize
python3 scripts/tokenize_messages_dataset.py
# → data/build/qwen3_30b_a3b_sft_f/{train,dev}.tokenized.jsonl

# 4. Train
python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --standalone \
    -m geak_agent_coder.sft.train \
    --config configs/sft/qwen3_coder_sft_f_2epoch.yaml
```

### Manifest

Schema: `geak_training_data_manifest_v2` (required by `assert_training_authorized`).
Tokenizer SHA256 pinned to `4dbc3bb2...` (Qwen3-Coder-30B-A3B-Instruct).
