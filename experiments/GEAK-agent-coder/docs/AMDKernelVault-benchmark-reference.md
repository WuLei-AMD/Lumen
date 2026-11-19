# AMDKernelVault Benchmark Reference

Reference: [AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU
Kernel Optimization](https://arxiv.org/abs/2609.12471) (Sep 2026)

Corpus: 62K+ validated HIP samples, 40K Triton kernels, 2.4K ROCm Libraries QA.
Model: Qwen3-8B, SFT + execution-aware RL on AMD GPUs.

---

## Benchmarks

### 1. PyTorch-to-HIP (translation)

- **Tasks**: 200, from KernelBench Levels 1 and 2
- **Input**: PyTorch reference implementation
- **Output**: HIP kernel that produces the same result, faster
- **Evaluation**: single-shot generation (no iteration)
- **Metrics**: Pass@1 (compile + correctness), Comp Acc, Corr Acc
- **Speedup basis**: vs PyTorch reference
- **Hardware**: AMD CDNA GPUs

| Model | Comp Acc | Corr Acc (Pass@1) |
|---|---|---|
| GPT-5 | 78.0% | 31.5% |
| Claude Sonnet 4 | 75.5% | 26.0% |
| Gemini 2.5 Pro | 71.5% | 25.0% |
| Qwen3-8B (base) | 50.0% | 14.5% |
| Qwen3-8B + SFT | 78.0% | 31.5% |
| Qwen3-8B + SFT + RL | 83.5% | 34.0% |

### 2. HIP-to-HIP (optimization)

- **Tasks**: 100 held-out HIP modules
- **Input**: existing HIP kernel
- **Output**: optimized HIP kernel
- **Evaluation**: single-shot generation
- **Metrics**: Comp Acc, Corr Acc, Speed@10 (avg speedup of best 10 correct samples)
- **Speedup basis**: vs input kernel

| Model | Comp Acc | Corr Acc | Speed@10 |
|---|---|---|---|
| GPT-5 | 89% | 87% | 1.32x |
| Claude Sonnet 4 | 93% | 67% | 1.11x |
| Gemini 2.5 Pro | 81% | 69% | 1.15x |
| Qwen3-8B (base) | 26% | 18% | 1.02x |
| Qwen3-8B + SFT | 86% | 70% | 1.06x |
| Qwen3-8B + SFT + RL | 90% | 84% | 1.14x |

### 3. TritonBench-G (Triton generation + optimization)

- **Tasks**: 184 kernels, 5 difficulty levels (D1 Easy → D5 Expert)
- **Input**: kernel task specification
- **Output**: optimized Triton kernel
- **Evaluation**: 3-iteration GEAK OptimAgent-v2 loop (model serves as
  generator, reflector, and optimizer under identical agent settings)
- **Metrics**: Corr@1, Corr@3, Comp@3, Speed
- **Speedup basis**: vs PyTorch reference (generation) or input kernel
  (optimization)
- **Hardware**: MI355X (inference), MI325X (training)

| Model | Corr@1 | Corr@3 | Comp@3 | Speed |
|---|---|---|---|---|
| GPT-5 | 3.8% | 15.2% | 28.8% | 1.05x |
| Claude Sonnet 4 | 7.6% | 29.8% | 46.7% | 1.34x |
| Gemini 2.5 Pro | 8.7% | 23.4% | 32.6% | 1.31x |
| Qwen3-8B (base) | 0.0% | 4.9% | 14.7% | 0.97x |
| Qwen3-8B + SFT | 4.3% | 10.3% | 27.7% | 1.19x |
| Qwen3-8B + SFT + RL | 12.0% | 33.2% | 44.0% | 1.19x |

Difficulty breakdown (SFT+RL vs Claude Sonnet 4):
- D1-D3: SFT+RL leads
- D4: Claude Sonnet 4 wins (18/84 vs 13/84)
- D5: unsolved by all models

### 4. ROCmBench (AMD-specific)

- **Tasks**: 31 AMD-specific kernels (rocBLAS/rocSOLVER related)
- **Evaluation**: 3-iteration GEAK OptimAgent-v2 (same protocol as
  TritonBench-G)
- **Metrics**: Corr@1, Corr@3, Comp@3, Speed, Pass@K (K=1-5, temp=1.0,
  5 samples each)

| Model | Corr@1 | Corr@3 | Comp@3 | Speed |
|---|---|---|---|---|
| GPT-5 | 16.13% | 29.03% | 35.48% | 1.31x |
| Claude Sonnet 4 | 19.35% | 35.48% | 67.74% | 1.82x |
| Gemini 2.5 Pro | 12.90% | 25.81% | 41.94% | 1.47x |
| Qwen3-8B (base) | 3.22% | 6.45% | 19.35% | 0.74x |
| Qwen3-8B + SFT + RL | 22.58% | 41.94% | 58.06% | 1.61x |

Pass@K on ROCmBench (31 kernels, 5 samples, temp=1.0):

| Model | Pass@1 | Pass@3 | Pass@5 |
|---|---|---|---|
| Qwen3-8B + SFT + RL | 27.10% | 38.71% | 41.94% |
| Claude Sonnet 4 | 25.81% | 36.77% | 38.71% |
| GPT-5 | 20.65% | 31.29% | 32.26% |
| Gemini 2.5 Pro | 17.42% | 29.03% | 35.48% |

---

## Metric Definitions

| Metric | Definition |
|---|---|
| Pass@1 | Compile + correctness pass on single generation |
| Comp@k | Fraction of tasks that compile by iteration k |
| Corr@k | Fraction of tasks that pass correctness by iteration k |
| Pass@K | Unbiased probability of ≥1 correct result from K samples |
| Speed | Mean speedup of correct samples vs baseline |
| Speed@10 | Mean speedup of best 10 correct samples vs baseline |

Speedup is only computed over correctness-passing samples.

---

## Evaluation Protocol

### Controlled comparison

All Triton model comparisons use identical GEAK OptimAgent-v2 configuration:
- `num_offsprings = 1`
- `max_iteration = 3`
- Same generator / reflector / optimizer role assignment
- No profiler feedback
- Only the model checkpoint changes between rows

### Train/test separation

- Evaluation cases excluded from training by source and task identifier
- HIP: 200 held-out KernelBench translation + 100 held-out HIP-to-HIP
  optimization
- Triton: 184-kernel TritonBench-G + 31-kernel ROCmBench
- Caveat: exact-match filtering cannot exclude every functional
  near-duplicate across the corpus

### Triton reproducibility

- Triton version: 3.3.0 (fixed)
- Corpus validation and profiling: MI325 and MI350
- Triton SFT and RL training: 32x MI325X
- Reported Triton inference: MI355X
- Corpus targets gfx942 (MI325X) and gfx950 (MI350/MI355X)

---

## Key Ablation Findings

### Data scale (PyTorch-to-HIP)

| Training data % | Comp Acc | Corr Acc |
|---|---|---|
| 0% (base) | 50.0% | 14.5% |
| 10% | 62.5% | 17.5% |
| 50% | 69.5% | 23.0% |
| 100% | 83.5% | 34.0% |

### Triton RL components

| Configuration | Corr@3 | Delta |
|---|---|---|
| Full system | 33.2% | — |
| Single-turn (T=1) | 20.1% | -13.1 |
| No SFT cold start | 8.5% | -24.7 |

Multi-turn is essential (+13.1%). SFT cold start is critical (+24.7%):
without SFT initialization, base model's ~15% initial success rate produces
extremely sparse RL gradients.

### Fixed agent scaffold ablation

| Model inside agent | Corr@3 |
|---|---|
| Qwen3-8B (base) | 4.9% |
| Qwen3-8B + SFT | 10.3% |
| Qwen3-8B + SFT + RL | 33.2% |

Same GEAK OptimAgent-v2 loop; only the model checkpoint changes.

---

## Common Failure Modes

- Incorrect tile/block size for unusual tensor shapes
- Synchronization errors in fused multi-operator kernels
- Numerically unstable reductions
- Inefficient memory layouts inherited from CUDA/NVIDIA assumptions
- D5 expert-level kernels remain unsolved by all models

---

## Resources

- Dataset: https://huggingface.co/datasets/amd/AIG-Datasets
- Code: https://github.com/AMD-AGI/hip_kernel_llm_lab
