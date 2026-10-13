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

---

# Phase 2 Data Contract

## Overview

Phase 2 extends the training data with three new capabilities for ISA-level
performance analysis, targeting the GEAK coder flow's `profile_engineer`,
`TechLead`, and `Engineer` roles:

1. **Bottleneck Analysis** (`bottleneck_analysis`) — static ISA census → structured JSON diagnosis
2. **Direction Generation** (`direction_generation`) — analysis → optimization direction + gates
3. **Kernel Code Generation** (`cold_start`, `direction_conditioned`) — OPUS HIP C++ patch generation

All three capabilities are served by the same fine-tuned model, called
three times in the GEAK flow to form a closed loop.

## Data Sources

### Performance Analysis (`perf_analysis_v1`)

Source: 9 optimization campaigns (D380–D450) from `perf-turns` dataset.
Each campaign contains verified base/new ISA `.s` files, kernel sources,
`kernel_edits/*.diff.md`, and Lead messages with bottleneck analysis + directions.

| Campaign | Bottleneck Subtype | Gain | Kernel Family |
|---|---|---|---|
| D380-S0-t16 | register_temporal_alias | -16.6% cold | kz_skv1_p1a_s0 |
| D393-S0-s2f | wave_occupancy_imbalance | -18.2% cold | kz_core_v2_s2f |
| D397-T128-fix | register_temporal_alias | +6.8% | gemm_a8w8_mxfp8_scale |
| D400-S0-ladder-t16 | vmem_wait_serialization | -3.99% cold | kz_skv1_p1a_s0 |
| D425-S0-ladder-s2f | vmem_wait_serialization | -3.28% cold | kz_core_v2_s2f |
| D438-SA-pad | lds_bank_conflict | +14.9% hot | kz_core_v2_s2f |
| D441-T128-N64-SAT | redundant_kernel_launch | +2.71% cold | gemm_a8w8_mxfp8_scale |
| D444-T128-fold | wasted_partial_tile_compute | +2.02% cold | gemm_a8w8_mxfp8_scale |
| D450-SA-preload-t16 | scale_path_memory_traffic | +3.9% | kz_skv1_p1a_s0 |

All campaigns are verified: base/new ISA SHA256-matched against snapshots (210 files, 0 bad).

### gfx950 Kernel Data (`Zhangdanyang/agent-kernel-sft-gfx950-triton-HIP`)

Extended training dataset:

| Component | Count | Source |
|---|---|---|
| Triton/HIP kernels | 2183 | GEAK agent runs + wave-300 |
| OPUS kernels | 875 | aiter production repo |
| FlyDSL kernels | 1000 | aiter production repo |
| General coding replay | 200 | SWE-Gym (140) + OpenCodeInstruct (60) |
| Performance analysis | 149 | perf-turns campaigns |
| **Total** | **4407** | |

### General Coding Replay (`general_coding_replay`)

- SWE-Gym (Apache-2.0): 140 samples, Python bugfix patches
- OpenCodeInstruct (CC-BY-4.0): 60 samples, coding tasks (average_test_score=1.0 only)
- Contamination-filtered: no HumanEval/MBPP/LiveCodeBench keywords

## New Task Type Schemas

### `bottleneck_analysis` (105 samples)

**Input:**

```json
{
  "kernel_source": "// full C++ kernel source...",
  "isa_census": {
    "symbols": {
      "<mangled_name>": {
        "instructions": 1241,
        "mfma": 160,
        "buffer_load": 92,
        "ds_read": 142,
        "ds_write": 48,
        "v_accvgpr_read": 64,
        "v_accvgpr_write": 32,
        "s_waitcnt": 28
      }
    },
    "resources": {"vgpr": 316, "sgpr": 96, "spills": 0}
  },
  "isa_excerpt": "// relevant ISA snippet...",
  "performance_gap": {
    "base_snapshot": "<snapshot-sha>",
    "gain_pct": 14.9,
    "gain_type": "hot",
    "shapes": ["#3 (128,7168,16384)"],
    "device": "gfx950"
  },
  "device": "MI355X / gfx950 / CDNA4, 256 CU, 160KB LDS/CU"
}
```

**Output:**

```json
{
  "bottleneck": "LDS-bound",
  "bottleneck_subtype": "lds_bank_conflict",
  "profiler_used": "isa_static_analysis",
  "device": "MI355X / gfx950",
  "key_metrics": {
    "vgpr_count": 316,
    "mfma_count": 160,
    "ds_read_count": 142,
    "spills": 0
  },
  "evidence": [
    {"type": "isa_pattern", "detail": "LDS rows stored at pitch=128..."},
    {"type": "census_diff", "detail": "Pad variant adds only 7 instructions..."}
  ],
  "top_opportunities": [
    "P0: Pad LDS row pitch from 128 to 132 bytes",
    "P1: Consider XOR swizzle"
  ],
  "shift_note": "Static ISA analysis, no PMC counters"
}
```

Seven trained bottleneck subtypes:
- `register_temporal_alias` — AGPR staging overlaps accumulator
- `vmem_wait_serialization` — vmcnt(0) blocks all pending transfers
- `wave_occupancy_imbalance` — VGPR pressure limits CU occupancy
- `lds_bank_conflict` — LDS pitch alignment causes bank aliasing
- `redundant_kernel_launch` — unnecessary standalone kernel dispatch
- `wasted_partial_tile_compute` — MFMA on partial tiles wastes compute
- `scale_path_memory_traffic` — scale vector LDS reads replaceable by VGPR preload

Sample tiers: T1 base diagnosis (9), T2 before/after (9), T3 sub-analysis (18),
T4 shape variants (27), T5 negative "balanced" (9), T6 synthetic (90 → 90 after cap),
T7 perturbation (40 → 40). Total after dedup: 105.

### `direction_generation` (36 samples)

**Input:**

```json
{
  "bottleneck_analysis": {"bottleneck": "...", "bottleneck_subtype": "...", ...},
  "kernel_source": "// base kernel source...",
  "optimization_history": [],
  "constraints": {
    "modifiable_files": ["kz_core_v2_s2f_AW4.cu"],
    "zero_spill": true,
    "lds_budget_bytes": 163840
  }
}
```

**Output:**

```json
{
  "direction": "S0 SA LDS bank de-aliasing: pad physical row pitch...",
  "gates": [
    "Census: only SA address symbols change",
    "LDS within budget",
    "0 spill",
    "Raw D byte-identical"
  ],
  "priority": "P0",
  "scope": "single_file_single_factor",
  "expected_evidence": "hot scenario: +14.9%"
}
```

Sources: 9 direct Lead message extractions + 27 shape-variant perturbations.

### Kernel Code Generation (4 `cold_start` + 4 `direction_conditioned`)

Uses the existing `geak_kernel_sft_v1` schema. Diffs are constructed from
`kernel_edits/*.diff.md` files (old→new pairs grouped by target file into
unified diff format). 4 unique `(session, turn)` pairs produce kernel diffs.

## Deduplication

- Internal: SHA256 of `task_type + "||" + assistant_content`
- Cross-split: SHA256 dedup against existing 4258 training samples
- Cross-campaign: session d59e994a turns shared across campaigns are
  deduplicated by `(session_id, turn_number)` for kernel_edits

## Three-Way Model Capability (GEAK Integration)

```
source + ISA  ──→  [1] Bottleneck Analysis  ──→  JSON (classification + evidence)
                          ↓
analysis + source ──→  [2] Direction Generation  ──→  direction + gates
                          ↓
direction + source ──→  [3] Kernel Code Generation  ──→  unified diff
```

| GEAK Role | Model Capability | Invocation |
|---|---|---|
| profile_engineer | Bottleneck Analysis | `staticAnalysisProfile()` |
| TechLead | Direction Generation | model as TechLead LLM backend |
| Engineer | Kernel Code Generation | model as Engineer LLM backend |

## Build Pipeline (Phase 2)

```bash
# 1. Generate performance analysis samples
python3 build_perf_analysis_dataset.py
# → appends to agent-kernel-sft-gfx950-triton-HIP/train.jsonl

# 2. Generate replay samples (if not already done)
python3 build_replay_dataset.py

# 3. Tokenize (same as Phase 1)
python3 scripts/tokenize_messages_dataset.py

# 4. Train
python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --standalone \
    -m geak_agent_coder.sft.train \
    --config configs/sft/qwen3_coder_sft_f_2epoch.yaml
```

## OPUS Sample Rebalancing

OPUS C++ kernel samples originally had case-specific aiter headers recursively
inlined, inflating many samples to 100K–515K characters. All inlined headers
have been **stripped and restored to correct `#include` references** so the model
learns proper header usage patterns (preventing hallucinated header content).
Large files are split into functional kernel units, each ≤ 32K characters.

### Design Rationale

- **Correct `#include` references**: All headers (both language-level and
  project-internal) are kept as `#include` directives. The model learns which
  headers to include for which operations, rather than memorizing inlined content.
- **Functional splitting**: Large files are split at function/struct boundaries
  (C++) or `def`/`class` boundaries (Python/FlyDSL), producing one kernel
  function or variant per sample.
- **Boilerplate removal**: Dispatch macros, pybind11 bindings, type switch
  tables, test code, and tuner code are classified and removed — they add no
  kernel implementation knowledge.

### Processing Pipeline

**Step 1 — Strip inlined headers:**
- Remove all `// === INLINED: xxx ===` ... `// === END INLINED: xxx ===` blocks
- Remove all `// === BEGIN INLINED: xxx ===` ... `// === END INLINED: xxx ===` blocks
- Restore each to its original `#include "xxx"` directive

**Step 2 — Split oversized samples (> 32K chars):**

For C++ files:
1. Extract header section (includes, pragmas, using, namespace, block comments)
2. Split body at blank-line boundaries (min 20 lines per section)
3. Group sections into chunks ≤ 32K chars (minus header budget)
4. Classify each chunk: `kernel` / `dispatch` / `pybind` / `macro` / `empty`
5. Keep only `kernel` chunks; prepend original header to each

For Python/FlyDSL files:
1. Extract header section (imports, docstrings, constants)
2. Split body at top-level `def` / `class` / `async def` boundaries
3. Group into chunks ≤ 32K chars
4. Classify each chunk: `kernel` / `test` / `tuner`
5. Keep only `kernel` chunks; prepend original header to each

**Step 3 — Deduplicate** by SHA256 of `(task_type || assistant_content)`.

### Chunk Classification Heuristics

| Category | Detection | Action |
|---|---|---|
| `pybind` | ≥5 hits of: `pybind11`, `py::arg`, `py::module`, `m.def(`, `TORCH_LIBRARY` | Remove |
| `dispatch` | ≥5 dispatch indicators (`case at::ScalarType::`, `AT_DISPATCH_`, `#define DISPATCH`, `TORCH_CHECK(false`) or macro_ratio > 0.5 with ≥2 indicators | Remove |
| `macro` | >70% of lines inside `#define` blocks | Remove |
| `test` | ≥2 of: `unittest`, `assert `, `def test_`, `pytest`, `class Test` | Remove |
| `tuner` | ≥2 of: `Tuner`, `tune_`, `argparse`, `if __name__` | Remove |
| `kernel` | None of the above | Keep |

### Results

| Metric | Before | After |
|---|---|---|
| Total samples | 3,670 | 3,975 |
| Max assistant chars | 515,793 | 32,000 |
| Samples > 32K | 280 (7.6%) | 0 |
| INLINED markers | 1,742 | 0 |

The increase in sample count (3,670 → 3,975) comes from splitting large
multi-function files into individual kernel samples.

### Build

```bash
python3 rebalance_opus_samples.py
# Processes: train.jsonl, artifacts/*/candidate/kernel.cu, heldout kernel.cu
# No external dependencies (aiter repo not needed — operates on already-built data)
```

## No Overlap with Benchmarks

Phase 2 performance analysis data covers operator `mxfp8_blockscale_gemm` only
(OPUS C++ kernels from perf-turns campaigns). Held-out benchmark tasks
(`agent-phase2-held-out-aiter`) cover 27 operators across 3 languages (Triton,
OPUS, FlyDSL) with 0 content overlap verified by SHA256.
