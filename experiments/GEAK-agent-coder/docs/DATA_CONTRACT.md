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
