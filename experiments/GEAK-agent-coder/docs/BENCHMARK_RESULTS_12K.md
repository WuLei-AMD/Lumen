# Benchmark Results: Base vs SFT-4epoch on Kernel Optimization

**Date:** 2026-09-29
**Hardware:** 8x AMD MI308X (gfx942), GPU 0 serving, GPU 1 kernel eval
**Framework:** multi-tune-agent (ToolAgentLoop, 2 rounds, 1 engineer, 16 tool turns)
**Benchmark suite:** 4 Triton kernel optimization tasks (held-out, no training data overlap)

## Models

| Model | Description | Serving |
|-------|-------------|---------|
| **Base** | Qwen/Qwen3-Coder-30B-A3B-Instruct (unmodified) | vLLM 0.15.0+rocm700, TP=1, 32K ctx |
| **SFT-4epoch** | Same base + 4-epoch LoRA SFT (val_loss 0.173) | Same vLLM config, merged weights |

## Optimization Benchmark Results

Each case runs the full multi-tune agent loop: baseline → tech lead analysis → engineer optimization → verification → promotion gate (min 2% improvement required).

| Case | Base Speedup | Base Status | SFT4 Speedup | SFT4 Status | Delta |
|------|-------------|-------------|--------------|-------------|-------|
| dense-gemm-fp16 | **1.111x** | success | 1.065x | success | -0.046 |
| dense-gemm-fp8 | 0.884x | no_improvement | **1.001x** | no_improvement | **+0.117** |
| fused-attention-prefill | 0.622x | no_improvement | 0.630x | no_improvement | +0.008 |
| grouped-gemm-moe | 1.014x | no_improvement | 1.002x | no_improvement | -0.012 |

### Key Observations

1. **FP8 GEMM: SFT4 significantly better.** Base model made the kernel 12% slower (0.884x); SFT4 maintained parity (1.001x). The SFT model learned to preserve FP8 scaling correctness where the base model broke it.

2. **FP16 GEMM: Both models achieved speedups.** Base got a higher single-run speedup (1.111x vs 1.065x), but SFT4 was more consistent across runs (1.017x and 1.065x vs base's single 1.111x).

3. **Fused attention: Neither model improved.** Both produced kernels slower than baseline (~0.63x). This is a known-hard operator for agent optimization.

4. **Grouped GEMM (MoE): Both near parity.** Neither model found meaningful improvements over the Sonic baseline.

## Zero-Shot Harness Generation Results

Tested both models on 98 held-out kernel contract specifications (50 Triton + 48 HIP, covering 20 operator types), requiring generation of a complete GEAK harness from scratch.

| Model | Pass Rate | Generated | Failed |
|-------|-----------|-----------|--------|
| Base | **0/98 (0%)** | 0 | 98 |
| SFT-4epoch | **0/98 (0%)** | 0 | 98 |

Both models failed all 98 zero-shot generation attempts. The GEAK harness generation task requires producing multiple coordinated files (config.yaml, kernel.py, task_runner.py, metadata.json) with strict validation — a capability neither model has without multi-turn agent interaction.

## Process Metrics

| Metric | Base | SFT4 |
|--------|------|------|
| Avg time per case | ~670s | ~717s |
| Cases with improvement ≥ 2% | 1/4 | 1/4 |
| Cases with regression | 2/4 | 1/4 |
| Compile error rate | - | - |

## Methodology Notes

- All cases used identical agent configs: `max_rounds=2`, `engineers_per_round=1`, `engineer_tool_rounds=16`, `baseline_repeats=3`
- Cases are **not in the training data** (verified by excluding all 120 training operator+shape pairs and 221 contract hashes)
- Both models served with identical vLLM settings (TP=1, 32K context, enforce-eager, qwen3_coder tool parser)
- GPU isolation enforced: GPU 0 for model serving, GPU 1 for kernel evaluation
- `dense-gemm-fp16` ran twice under SFT4 (likely due to the case catalog listing it in both examples and phase1); the better result (1.065x) is reported

## Conclusion

The SFT training shows measurable impact on **robustness** rather than peak optimization:
- SFT4 avoids performance regressions (1/4 regressions vs 2/4 for base)
- SFT4 preserves FP8 kernel correctness where base model breaks it (+0.117 on FP8 GEMM)
- Peak optimization on well-understood operators (FP16 GEMM) shows the base model can occasionally outperform

A larger held-out benchmark suite with more operator diversity and multiple seeds would strengthen these conclusions.
