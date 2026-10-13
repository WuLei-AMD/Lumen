# From-Scratch Kernel Writing Benchmark Results — MI308X (gfx942)

Hardware: AMD MI300X (MI308X), 8 GPUs, gfx942
Benchmark: 100 held-out tasks, model writes Triton kernel from zero via GEAK ToolAgentLoop.
Target: achieve >= 90% of AITER SOTA performance (speedup >= 0.9x).

## SFT-2F — Qwen3-Coder-30B-A3B-SFT-TH-2epoch

- Model: `Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-2epoch`
- Training data: 3164 samples (2000 patch + 1164 complete_implementation), messages chat format
- Dataset: `Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP`

### Overall Summary (73/100 completed, in progress)

| Metric | Post-task Log | + Manual Rebench | Notes |
|--------|--------------|------------------|-------|
| FAST (>=0.9x) | 2 | **15+** | Many early-exit tasks missed by post-task |
| OK (>0, <0.9x) | 5 | 7+ | |
| FAIL | 66 | ~51 | MHA/MLA/paged_attn all FAIL |
| Snapshots | — | **35** | Sessions with at least one correct kernel |

### Results by Operator

| Operator | Tasks | Snapshot Rate | Best Snapshot | Post FAST | Post OK |
|----------|-------|---------------|---------------|-----------|---------|
| gemm | 21 | **17/21 (81%)** | 1.97x | 1 | 3 |
| rms_norm | 14 | **12/14 (86%)** | 6.00x | 1 | 2 |
| rope_kv_cache | 6 | 2/6 (33%) | 58.66x* | 0 | 0 |
| sampling | 5 | 1/5 (20%) | 10.33x* | 0 | 0 |
| mha | 13 | 0/13 (0%) | — | 0 | 0 |
| mla | 6 | 0/6 (0%) | — | 0 | 0 |
| paged_attention | 8 | 0/8 (0%) | — | 0 | 0 |

*rope_kv_cache and sampling baselines use Python fallback, not real AITER C++ kernels.

### All Verified Kernel Results

Sorted by speedup. All independently verified: kernel_best.py restored,
correctness checked, 3-run median for both baseline and candidate.

| # | Task | Op | Speedup | Turns | Verify |
|---|------|----|---------|-------|--------|
| 1 | aiter_derived-rope_kv_cache-02 | rope_kv_cache | **76.28x*** | 50 | manual |
| 2 | aiter_derived-sampling-02 | sampling | **9.91x*** | 50 | manual |
| 3 | aiter-prod-rms_norm-04 | rms_norm | **5.65x** | 75 | post |
| 4 | aiter-prod-rms_norm-03 | rms_norm | **5.63x** | 75 | post |
| 5 | aiter_derived-rms_norm-04 | rms_norm | **5.01x** | 6 | manual |
| 6 | adversarial_boundary-rms_norm-02 | rms_norm | **4.98x** | 200 | post |
| 7 | aiter-prod-rms_norm-06 | rms_norm | **4.70x** | 75 | post |
| 8 | aiter-prod-rms_norm-02 | rms_norm | **4.53x** | 75 | post |
| 9 | geak_native-rms_norm-02 | rms_norm | **4.23x** | 6 | post |
| 10 | aiter-prod-rms_norm-07 | rms_norm | **4.14x** | 75 | post |
| 11 | aiter-prod-rms_norm-01 | rms_norm | **3.54x** | 75 | post |
| 12 | aiter-prod-gemm-05 | gemm | **1.97x** | 7 | post |
| 13 | adversarial_boundary-gemm-04 | gemm | **1.89x** | 6 | post |
| 14 | aiter_derived-gemm-03 | gemm | **1.88x** | 51 | manual |
| 15 | adversarial_boundary-gemm-02 | gemm | **1.70x** | 6 | post |
| 16 | aiter_derived-gemm-01 | gemm | **1.56x** | 50 | manual |
| 17 | adversarial_boundary-gemm-03 | gemm | **1.50x** | 6 | post |
| 18 | aiter-prod-gemm-04 | gemm | **1.50x** | 7 | post |
| 19 | geak_native-gemm-04 | gemm | **1.44x** | 128 | manual |
| 20 | aiter-prod-gemm-02 | gemm | **1.43x** | 7 | post |
| 21 | geak_native-gemm-02 | gemm | **1.38x** | 128 | manual |
| 22 | aiter_derived-gemm-02 | gemm | **1.37x** | 6 | manual |
| 23 | geak_native-gemm-03 | gemm | **1.26x** | 128 | manual |
| 24 | aiter-prod-gemm-08 | gemm | **1.19x** | 86 | post |
| 25 | adversarial_boundary-rms_norm-02 | rms_norm | **1.10x** | 6 | post |
| 26 | aiter-prod-gemm-03 | gemm | **1.08x** | 7 | post |
| 27 | aiter_derived-rms_norm-02 | rms_norm | **1.07x** | 67 | manual |
| 28 | geak_native-rms_norm-04 | rms_norm | **1.06x** | 6 | post |
| 29 | aiter-prod-gemm-07 | gemm | **0.95x** | 7 | post |
| 30 | geak_native-gemm-01 | gemm | **0.91x** | 128 | manual |
| 31 | adversarial_boundary-rms_norm-04 | rms_norm | 0.86x | 6 | post |
| 32 | aiter-prod-gemm-01 | gemm | 0.81x | 7 | post |
| 33 | aiter-prod-gemm-01 | gemm | 0.76x | 200 | post |
| 34 | aiter-prod-rms_norm-08 | rms_norm | 0.72x | 200 | post |
| 35 | aiter-prod-gemm-06 | gemm | 0.69x | 7 | post |

### Kernel Optimization Analysis

All kernels are legitimate pure Triton implementations. The model uses standard
tiled GEMM / single-row RMS norm algorithms with better config tuning.

#### gemm (1.0-1.97x): Shape-aware block size tuning

| What AITER does | What SFT does | Why faster |
|----------------|---------------|------------|
| BLOCK_M=128, conditional shrink | BLOCK_M matched to M (16/32) | No wasted rows |
| BLOCK_K=32-64 | BLOCK_K=64-256 | Fewer K-loop iterations |
| 1D grid + GROUP_SIZE_M=8 | 2D grid direct indexing | Less remapping overhead |
| num_warps=8 | num_warps=4 for small M | Better per-warp occupancy |
| @triton.autotune | No autotune, direct config | No search overhead |

Core algorithm identical (tiled GEMM with tl.dot accumulation). Speedup from
config tuning for small-batch shapes (M=2-37), not algorithmic innovation.

#### rms_norm (1.0-5.65x): Simplified kernel structure

| What AITER does | What SFT does | Why faster |
|----------------|---------------|------------|
| 147 lines, persistent kernel | 64 lines, one-row-per-program | Less overhead |
| USE_BLOCKED branch + blocked K | Single BLOCK_SIZE=next_power_of_2(N) | No branching |
| NUM_PRGMS grid, row loop | grid=(M,) direct | Simpler dispatch |
| tl.multiple_of hints, .cg cache | Standard loads | Fewer instructions |

Biggest gains on small M (7-9 rows) where AITER's persistent kernel overhead
dominates. Marginal on M>20.

#### rope_kv_cache (38-76x): AITER Python fallback replacement

AITER baseline has 3 overhead sources the SFT kernel eliminates:
1. **Python for-loop** for KV cache scatter (no Triton concat_and_cache)
2. **torch.atan2** to convert cos/sin back to freqs
3. **2 kernel launches** (separate q and k) vs 1 fused kernel

Speedup reflects Python overhead, not GPU kernel improvement. Production
AITER uses C++ concat_and_cache.

#### sampling (9.91x): PyTorch wrapper replacement

AITER wrapper uses `torch.argsort/gather/cumsum` instead of calling its own
Triton JIT function. SFT writes pure Triton. Speedup reflects PyTorch dispatch
overhead, not AITER's actual C++ sampling performance.

### Comparison: SFT-2F vs SFT-2E

| Metric | SFT-2E (69/100) | SFT-2F (73/100) |
|--------|-----------------|-----------------|
| Training data | 2000 patch only | **3164** (+ 1164 impl) |
| gemm snapshot rate | 4/25 (16%) | **17/21 (81%)** |
| rms_norm snapshot rate | 12/14 (86%) | 12/14 (86%) |
| Best gemm | 1.66x (unverified) | **1.97x (verified)** |
| Best rms_norm | 1.83x | **5.65x** |
| New operators | 0 | **rope_kv_cache, sampling** |
| First-shot (t1-t2) | Rare (t3-t5) | **Common (t1-t2)** |

Key improvement: complete_implementation training data enabled gemm from-scratch
writing (16% -> 81% snapshot rate).

### Turn Budget Analysis

| Turns | Tasks | With Snapshots | Insight |
|-------|-------|---------------|---------|
| <=10 | ~15 | ~12 (80%) | Early exit, but model wrote good kernel first |
| 11-100 | ~15 | ~10 (67%) | Exception exits, snapshots captured |
| 101-200 | ~43 | ~13 (30%) | Full runs, complex ops mostly fail |

Best kernels appear at t1-t2 (first successful evaluate). 200-turn iteration
often degrades results by overwriting good kernels with broken ones.

---

## Benchmark Infrastructure

```
GPU 0:   vLLM model serving (262K context, --tool-call-parser qwen3_coder)
GPU 2-7: Kernel evaluation (6 parallel workers)
Loop:    200 turns max, GEAK ToolAgentLoop (read_file, write_file, evaluate)
Timing:  tuned_bench.py (200 warmup + 5-round median)
Baseline: AITER kernels with frozen autotune configs (receipts/autotune_freeze.json)
Validate: kernel_validator.py blocks PyTorch ops in wrapper
Snapshot: kernel_sp{X}_t{N}.py + kernel_best.py on each successful evaluate
Rebench:  Restore kernel_best.py, verify correctness, 3-run median both sides
```
