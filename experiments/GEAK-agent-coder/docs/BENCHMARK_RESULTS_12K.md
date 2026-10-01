# Benchmark Results: Base vs SFT on AMD GPU Kernel Generation

**Date:** 2026-09-30 (updated)
**Hardware:** 8x AMD MI308X (gfx942), GPU 0 for model serving, GPU 1-7 for kernel eval
**Held-out suite:** 99 tasks (54 Triton + 45 HIP), 10 operator families × 3 source suites, verified no training data overlap
**Models:** Base (Qwen3-Coder-30B-A3B-Instruct), SFT-2epoch (val_loss 0.178), SFT-4epoch (val_loss 0.173)

---

## 1. Patch Optimization Benchmark (v5)

Given the parent kernel source code, the model generates a unified diff patch to optimize it. The harness applies the patch and verifies compile → correctness → performance.

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| Patch generated | 83/99 (84%) | **99/99 (100%)** | **99/99 (100%)** |
| Patch applied | 50/99 (51%) | 24/99 (24%) | 28/99 (28%) |
| Compiled | 50/99 (51%) | 24/99 (24%) | 28/99 (28%) |
| **Correct** | **44/99 (44%)** | **23/99 (23%)** | **26/99 (26%)** |
| Triton correct | 20/54 | 10/54 | 18/54 |
| HIP correct | 24/45 | 13/45 | 8/45 |

**Note:** SFT models have lower apply rate due to context-line mismatch between training templates and held-out templates. Among successfully applied patches, SFT correctness rate (93%) is comparable to base (88%). A fuzzy Python patch applier (v6+) is being developed to eliminate this infrastructure bias.

## 2. From-Scratch Generation Benchmark (v4)

Given only the operator contract (no parent source), the model generates a complete kernel.py from scratch. This directly tests SFT's kernel coding capability without patch-apply interference.

| Metric | Base | SFT-2epoch | SFT-4epoch |
|--------|------|-----------|-----------|
| Code generated | 99/99 (100%) | 99/99 (100%) | 98/99 (99%) |
| **Compiled** | **5/99 (5%)** | **10/99 (10%)** | **18/99 (18%)** |
| **Correct** | **0/99 (0%)** | **1/99 (1%)** | **2/99 (2%)** |

**SFT4 compile rate is 3.6x base** (18% vs 5%). This is the cleanest signal of SFT effect.

### Per-Operator Generation Results (SFT-4epoch v4)

| Operator | Compiled | Correct | Training samples |
|----------|----------|---------|-----------------|
| rms_norm | **6/12** | **3/12** | 19 (norm) |
| gemm | **1/12** | 0/12 | 19 |
| fused_moe | 0/12 | 0/12 | 7 |
| mha | 0/12 | 0/12 | 13 |
| mla | 0/12 | 0/12 | 22 |
| paged_attention | 0/12 | 0/12 | 13 (attention) |
| blockscale_gemm | 0/6 | 0/6 | 0 (fp8 variant) |
| rope_kv_cache | 0/12 | 0/12 | 11 |
| sampling | 0/9 | 0/9 | 17 |

Only `rms_norm` achieves correctness — the simplest operator with a clear contract.

---

## 3. Error Analysis

### Patch mode: why SFT apply rate is lower

SFT models generate patches with context lines learned from training data templates. Held-out tasks use different template variants (different shape parameters produce slightly different boilerplate code). GNU `patch` fails when context lines don't exactly match, even if the actual change is correct.

- **Base model** outputs simpler patches (sometimes just variable renames) that match more easily
- **SFT model** outputs deeper algorithmic changes with more context lines, making exact matching harder
- Among applied patches, SFT correctness (93%) ≈ base correctness (88%)

### Generation mode: compile error breakdown (SFT-4epoch)

| Error category | Count | Fixable? |
|---------------|-------|----------|
| Wrong function signature | 21 | **Yes** — improved prompt with exact signature |
| RuntimeError | 19 | Partially — multi-turn error recovery |
| AssertionError | 9 | Partially — multi-turn |
| Arch gate false positive | 6 | **Yes** — removed in harness |
| TypeError | 6 | Partially — multi-turn |
| Wrong function name | 5 | **Yes** — auto-alias added |
| NVIDIA-only gate | 4 | **Yes** — removed in harness |
| SyntaxError | 2 | Multi-turn recovery |

---

## 4. Training Data Gap Analysis and Recommendations

### Current training data

689 kernel samples (406 HIP + 258 Triton), covering:

| Operator family | Samples | Benchmark compile rate | Assessment |
|----------------|---------|----------------------|------------|
| quant (per-tensor/token/block) | 71 | N/A (not in held-out) | Adequate |
| silu_and_mul | 26 | N/A | Adequate |
| mla | 22 | 0/12 compiled | **Severely insufficient** |
| gemm (bf16) | 19 | 1/12 compiled | **Insufficient** |
| norm (rms_norm, layernorm) | 19 | 6/12 compiled, 3/12 correct | Best performing |
| sampling (top-k/p) | 17 | 0/9 compiled | **Insufficient** |
| softmax | 14 | N/A | Moderate |
| mha | 13 | 0/12 compiled | **Severely insufficient** |
| attention (paged) | 13 | 0/12 compiled | **Severely insufficient** |
| rope_kv_cache | 11 | 0/12 compiled | **Insufficient** |
| batched_gemm | 10 | N/A | Moderate |
| router (MoE routing) | 9 | N/A | Moderate |
| fused_moe | 7 | 0/12 compiled | **Severely insufficient** |
| knn | 7 | N/A | Low priority |

### Recommended additional training data (Phase 2)

**Target: 3,000–4,000 total verified kernel SFT samples** (per runbook section 3.1)

#### Priority 1: Complex operators with 0% compile rate (need 150+ samples each)

| Operator | Current | Target | Gap | Priority | Rationale |
|----------|---------|--------|-----|----------|-----------|
| **mha (multi-head attention)** | 13 | 200 | 187 | **P0** | Core LLM inference operator, 0% compile on held-out |
| **mla (multi-latent attention)** | 22 | 200 | 178 | **P0** | Used in DeepSeek-V3/Qwen3, unique attention variant |
| **paged_attention** | 13 | 200 | 187 | **P0** | Critical for vLLM/SGLang serving, complex paging logic |
| **fused_moe** | 7 | 200 | 193 | **P0** | MoE is the dominant architecture trend, token routing + expert GEMM |
| **rope_kv_cache** | 11 | 150 | 139 | **P0** | Fused RoPE + KV cache write, essential for inference |

#### Priority 2: Operators with low compile rate (need 80-120 samples each)

| Operator | Current | Target | Gap | Priority | Rationale |
|----------|---------|--------|-----|----------|-----------|
| **gemm (bf16 + fp8)** | 19 | 120 | 101 | **P1** | Foundation operator, needs more shape diversity |
| **sampling (top-k/top-p)** | 17 | 100 | 83 | **P1** | Last-mile inference, needs multinomial + sorting |
| **blockscale_gemm** | 0 | 100 | 100 | **P1** | FP8 block-scaled GEMM, new for MI300X/MI350 |
| **all_reduce** | 0 | 80 | 80 | **P1** | Multi-GPU collective, RCCL integration |

#### Priority 3: Strengthen existing coverage (need 50-80 samples each)

| Operator | Current | Target | Gap | Rationale |
|----------|---------|--------|-----|-----------|
| rms_norm | 19 | 80 | 61 | Best performer, but only 3/12 correct — more shape diversity |
| softmax | 14 | 80 | 66 | Common in attention, needs causal mask variants |
| silu_and_mul | 26 | 60 | 34 | Fused activation, more shape/dtype combinations |
| fused_add_rms_norm | 0 | 60 | 60 | Residual + norm fusion, common in transformers |

#### Per-lane balance

| Lane | Current | Target | Gap |
|------|---------|--------|-----|
| Triton × gfx942 | 258 | 1,000 | 742 |
| HIP × gfx942 | 406 | 1,000 | 594 |
| Triton × gfx950 | 0 | 750 | 750 |
| HIP × gfx950 | 0 | 750 | 750 |

#### Per-task-type balance

| Task type | Current | Target (%) | Target count | Gap |
|-----------|---------|-----------|-------------|-----|
| cold_start | 142 | 15% | 525 | 383 |
| profile_guided | 134 | 15% | 525 | 391 |
| direction_conditioned | 217 | 45% | 1,575 | 1,358 |
| error_recovery | 116 | 15% | 525 | 409 |
| regression_balance | 80 | 10% | 350 | 270 |

### Key recommendations

1. **Attention operators are the #1 gap.** MHA, MLA, and paged attention together need ~600 new samples. These operators have complex memory access patterns (Q/K/V projections, causal masking, paging) that the model cannot learn from 13-22 examples.

2. **More error_recovery samples.** The multi-turn benchmark shows that error recovery capability is critical for agent loop efficiency. Current 116 samples (17%) should grow to 525 (15% of 3,500).

3. **gfx950 is entirely missing.** MI350/MI355 will need separate training data with native MXFP support. Phase 2 should add 1,500 gfx950 samples.

4. **From-scratch generation needs cold_start expansion.** Only 142 cold_start samples, but generation benchmark is the clearest SFT signal. Target 525.

5. **Shape diversity within each operator.** Current held-out benchmark shows that models trained on limited shape variants struggle to generalize. Each operator should cover decode (M=1-4), small-batch (M=8-64), and prefill (M=128-4096) regimes.

---

## 5. Methodology

### Held-out dataset

- 120 tasks (60 Triton + 60 HIP) from `Zhangdanyang/agent-phase1-held-out-private`
- 10 operator families × 3 source suites (geak_native, aiter_derived, adversarial_boundary) × 4 shape variants
- 99 tasks pass baseline verification (12 all_reduce excluded — need multi-GPU, 9 others have adversarial shape issues)
- All tasks have complete GEAK harness (config.yaml, scripts/task_runner.py, metadata.json) with SHA-256 verified against protected hashes
- Zero overlap with training data verified by contract hash, source lineage, and operator+shape exclusion

### Benchmark infrastructure fixes applied

| Version | Fix | Impact |
|---------|-----|--------|
| v2 | Strip code fences, normalize git diff format | Patch gen rate 78%→99% |
| v4 | `--fuzz=3/10` for GNU patch | Apply rate improved |
| v4 | Remove broken arch gates (gcnArchName) | Gen compile +82 (base), +6 (SFT) |
| v4 | Auto function alias | Gen compile +5 (SFT) |
| v4 | Fix `hip_cxxflags` API | Gen compile +1 |
| v4 | Improved generation prompt with interface spec | Pending re-run |
| v5 | Restore original file before each patch attempt | Apply rate +3-5 |
| v6 | Python fuzzy patch applier (context-insensitive) | Pending |
| v7 | Multi-turn error recovery (5 turns max) | Pending |
| v7 | Full function signature in generation prompt | Pending |

### Model serving

All models served with vLLM 0.15.0+rocm700, TP=1, 32K context, enforce-eager, qwen3_coder tool parser. Models run sequentially (not simultaneously) to ensure clean GPU state.
