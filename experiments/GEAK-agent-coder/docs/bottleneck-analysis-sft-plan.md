# Bottleneck Analysis & Kernel Optimization SFT Plan

## Overview

Train a Qwen3-Coder-30B-A3B LoRA adapter with three additional capabilities
sourced from expert kernel optimization data on gfx950:

1. **Bottleneck analysis** — static ISA-level bottleneck classification
2. **Direction generation** — optimization direction + acceptance gates
3. **Kernel code generation** — gfx950 HIP kernel patches (cold-start + directed)

All three capabilities serve the GEAK coder flow: the same fine-tuned model
acts as profile analyst, optimization architect, and code engineer.

## Training Data

### Task types and volumes

| Task | Approx. samples | Schema |
|---|---|---|
| Bottleneck analysis | ~200 | `geak_bottleneck_sft_v1` |
| Direction generation | ~50–60 | `geak_bottleneck_sft_v1` |
| Kernel generation (cold-start) | ~80–110 | `geak_kernel_sft_v1` |
| Kernel generation (directed) | ~7 | `geak_kernel_sft_v1` |
| Existing GEAK kernel coding (gfx942) | 2,000 | `geak_kernel_sft_v1` |
| **Total** | **~2,340–2,380** |

New data represents ~14–16% of the combined training set.

### Bottleneck analysis

- **Input**: kernel source + ISA census (per-symbol instruction counts, resource
  usage) + performance gap description
- **Output**: structured JSON extending GEAK's `PROFILE_SCHEMA` — bottleneck
  classification, evidence chain, ranked optimization opportunities
- Covers multiple bottleneck categories aligned with GEAK's existing taxonomy
- Samples constructed from expert-verified optimization campaigns and augmented
  via knowledge-base synthesis and template perturbation

### Direction generation

- **Input**: bottleneck analysis result + kernel source + constraints
- **Output**: specific optimization direction with verifiable acceptance gates
- Sourced from expert optimization architect directives
- Enables the model to bridge analysis and code generation

### Kernel code generation

- **Cold-start mode**: model receives only source code (no direction) and must
  independently identify and apply the optimization. Expert knowledge is encoded
  in model weights (IP protection — not exposed via external Skill files).
- **Direction-conditioned mode**: model follows an explicit optimization direction.
- All samples verified at the campaign level (compile, correctness, benchmark).

## Training Pipeline Changes

| Component | Change |
|---|---|
| `contracts.py` | Register `geak_bottleneck_sft_v1` schema + validation |
| `formatting.py` | Add routing for bottleneck/direction task types with dedicated system prompts |
| `configs/data/` | New data config YAML with additional sources |
| `sampling.py` | No change (automatic stratum assignment) |

## GEAK Flow Changes

### Decision boundaries

The upgraded flow adds a fast path before the existing TechLead-directed loop.
Three decision boundaries determine when to use the model's embedded knowledge
vs external tools:

**Boundary 1 — ISA analysis vs GPU profiler:**
Use the model's static ISA analysis when the bottleneck is structural and the
model reports high-confidence evidence. Fall back to GPU profiling when the
bottleneck involves runtime behavior (bandwidth utilization, cache, scheduling)
or evidence is insufficient.

**Boundary 2 — Cold-start vs TechLead direction:**
Attempt cold-start when the model identifies a single, well-understood
bottleneck with a single-factor fix and it's the first optimization round.
Require TechLead direction for multi-bottleneck scenarios, multi-file changes,
or after a cold-start failure.

**Boundary 3 — Self-retry vs TechLead fallback:**
Allow one self-directed retry when the failure is a "writing error" (compile or
correctness). Fall back to TechLead when the failure is a "direction error"
(patch passes but no speedup — wrong optimization approach).

### Flow summary

```
Benchmark → Model ISA Analysis → Confidence Check
                                   /            \
                              confident      not confident
                                 ↓                ↓
                          Cold-start?      GPU Profiler (merge)
                           /       \             ↓
                        yes        no       TechLead Direction
                         ↓          ↓            ↓
                    Cold-start   TechLead     Engineer
                         ↓       flow            ↓
                      Verify     ...          Verify
                      /    \
                   pass    fail
                    ↓        ↓
                 done    write error? → self-retry (1x)
                         direction error? → TechLead fallback
```

The existing TechLead + Engineer flow is fully preserved as fallback.

### GEAK backend extension

A new task directory provides the oracle (baseline, correctness harness,
contract) for gfx950 OPUS kernels. The GEAK framework is task-dir-driven
and does not require core orchestration changes.

## Verification

1. **Data**: schema roundtrip, patch-applies, evidence consistency, token budget
2. **Model**: bottleneck classification accuracy on held-out campaigns, direction
   quality, cold-start code quality, gfx942 coding regression check
3. **Integration**: task-dir smoke test, static analysis endpoint test, decision
   boundary tests, end-to-end GEAK run on gfx950

## Execution Phases

| Phase | Duration | Requirements |
|---|---|---|
| Data construction | ~4 days | CPU only |
| Training pipeline adaptation | ~1 day | CPU only |
| GEAK backend extension | ~1 week | gfx950 machine |
| Training + evaluation | varies | 8×MI308X + gfx950 |
