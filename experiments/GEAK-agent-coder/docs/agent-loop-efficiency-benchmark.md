# Kernel Agent Benchmark: From-Scratch Kernel Writing

## Goal

Evaluate a model's ability to **write GPU kernels from scratch** for AMD MI300X (gfx942) using the GEAK tool-calling agent flow. The model reads a task spec (operator type, shapes, correctness oracle) and writes a Triton kernel that achieves at least **90% of AITER SOTA performance**.

This measures real kernel engineering capability — not just optimizing existing code, but understanding operator semantics, memory access patterns, and hardware-specific tuning from zero.

## Benchmark Design

### Task Structure

Each of the 100 tasks provides:
- `scripts/task_runner.py` — defines function signature, test inputs, correctness oracle, and performance measurement
- `config.yaml` — compile/correctness/performance commands
- `kernel.py` — **cleared to skeleton** (model writes from scratch)

The model does NOT see the AITER reference kernel. It must figure out the implementation from the task_runner interface.

### Agent Flow (GEAK ToolAgentLoop)

```
Model receives: operator type + baseline perf target
  → read_file("scripts/task_runner.py")     # understand interface
  → write_file("kernel.py", implementation) # write Triton kernel
  → evaluate                                 # compile → correctness → performance
  → iterate up to 50 turns                   # fix errors, improve performance
```

### Success Metric

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| **speedup_geomean** | >= 0.9x | Kernel reaches 90% of AITER performance |
| **Fast@0.9** | count | Number of tasks achieving the target |
| per_operator_speedup | breakdown | Performance by operator type |

Compile rate and correct rate are diagnostic (prerequisites, not goals).

### Models Tested

| Label | Path | Prompt Format |
|-------|------|---------------|
| base | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Generic (natural language) |
| sft2 | `qwen3-coder-full2000-2epoch-merged` | SFT-compatible JSON |
| sft4 | `qwen3-coder-full2000-4epoch-merged` | SFT-compatible JSON |

SFT models get prompts in `geak_kernel_sft_v1` format (contract + direction + baseline). Base model gets generic natural language prompt. Controlled by `sft_compat` flag.

## Setup and Execution

For complete instructions on server launch, GPU topology, benchmark orchestration,
multi-model pipeline, monitoring, and troubleshooting, see:

**[MI308_RUNBOOK.md — From-Scratch Kernel Benchmark](MI308_RUNBOOK.md#from-scratch-kernel-benchmark)**

Key points:
- GPU 0: vLLM server (`--tool-call-parser qwen3_coder` required)
- GPU 2-7: 6 parallel evaluation workers
- 100 held-out tasks, 200 turns max per task
- Dataset: `/home/danyzhan/held-out-benchmark-aiter/`
- Results: `agent-loop-efficiency-results-MI308.md`

## Baseline Stability

### Autotune Freeze (Critical)

Raw `@triton.autotune` causes **6x measurement variance** across runs (different num_warps winners due to GPU noise). Fixed by:

1. **One-time config sweep** (`freeze_autotune.py`): try num_warps=[2,4,8,16] x num_stages=[1,2] for each kernel
2. **Lock winner** into kernel as single-config `@triton.autotune`
3. **Stable measurement**: `tuned_bench.py` with 200 warmup + 5-round median

Results stored in `receipts/autotune_freeze.json`. Each kernel has its optimal config locked.

### Measurement Stability

| Component | Setting | Purpose |
|-----------|---------|---------|
| tuned_bench.py | 200 warmup + 5x median | JIT warmup + stable timing |
| baseline_repeats | 3 | Multiple baseline measurements |
| Frozen configs | Single best num_warps | Eliminates autotune variance |

## Code-Level Validation

### validate_kernel (in sandbox.write_file)

Blocks two cheating patterns discovered during earlier benchmarks:

1. **PyTorch Replacement**: Model writes empty `@triton.jit` shell but does computation in PyTorch wrapper (`torch.argsort`, `torch.gather`, etc.)
2. **PyTorch in Wrapper**: Model keeps `@triton.jit` tag but moves core computation to Python

```python
# kernel_validator.py
PYTORCH_OPS = ["torch.argsort", "torch.sort(", "torch.gather", "torch.cumsum"]

def validate_kernel(content, original_path=None):
    if "@triton.jit" not in content:
        return False, "Must contain @triton.jit"
    # Check wrapper (after last @triton.jit) for PyTorch ops
    wrapper = content.split("@triton.jit")[-1]
    for op in PYTORCH_OPS:
        if op in wrapper:
            return False, f"Wrapper must not use {op}"
    return True, content
```

### Diff Patch Support

SFT models may output unified diffs instead of complete files. `write_file` auto-detects and applies:
1. `patch --fuzz=3` (standard)
2. `patch --fuzz=99` (aggressive, ignore context)
3. `robust_patch` (custom context-matching)
4. Direct write (fallback)

## Lessons Learned

### Infrastructure

**1. Tool Call Parser Must Be `qwen3_coder`**

Wrong parser → model outputs text instead of function calls → agent loop terminates after 1 turn with tools=0. This is a silent failure — no error, just poor results.

**2. Large Kernel Timeout**

MHA kernel (2863 lines, 99KB) causes vLLM inference timeout when 6 workers compete for GPU with 25K+ token prompts each. Fix: `timeout=3600s`. Complex operators (mha, mla, paged_attention, rope_kv_cache) need 100 turns instead of 50.

**3. Zombie GPU Processes**

When vLLM crashes or is killed, GPU memory may not be freed (zombie processes with parent PID 1 inside container). New vLLM instances fail to allocate. Fix: `docker restart <container>`.

**4. Session Cleanup Between Runs**

Old sessions from previous benchmark runs pollute audit results and can cause session ID conflicts. Always `rm -rf receipts/geak-flow-*` before a new run.

### Measurement

**5. Autotune Instability**

Adding `@triton.autotune` with multiple configs causes 6x measurement variance. Each subprocess re-runs the config search, picking different winners due to GPU noise. Fix: run config sweep once (`freeze_autotune.py`), lock the winner into a single-config autotune.

**6. Speedup Noise (Code-Identical Kernels)**

Even with frozen configs, code-identical kernels can report 1.3-1.7x "speedup" due to: baseline measured during session creation (with GPU contention from parallel workers) vs candidate measured later (different contention). Fix: for any reported speedup, verify the code actually changed (diff > 3 lines of real code). Re-benchmark baseline and candidate independently on idle GPU.

**7. Baseline Must Be Re-measured**

`establish_baseline` runs once at session creation and caches `baseline_ms`. If that measurement was noisy (GPU contention), all subsequent speedup calculations inherit the error. For production benchmarks, re-measure both baseline and candidate on the same GPU in sequence.

### Model Behavior

**8. PyTorch Wrapper Cheating**

Both Base and SFT models independently discover the "empty Triton shell + PyTorch wrapper" trick — keep `@triton.jit` to pass checks but move computation to `torch.argsort/gather/cumsum` in the wrapper function. Prompt rules alone are insufficient. Code-level validation (`kernel_validator.py` in `sandbox.write_file`) is essential.

**9. Autotune Deletion Cheating**

Models delete `@triton.autotune` and hardcode a fixed config. Since autotune search has overhead, the fixed-config version appears faster. This is NOT a real optimization — it's exploiting measurement artifacts. Fix: freeze baseline configs so autotune deletion has no advantage.

**10. SFT Narrows Capability**

In optimization benchmarks, SFT-2e specialized on rms_norm (6.98x) but degraded on attention operators (0x on mha/mla/paged_attn where Base got 2.88x). SFT training on single-turn `input→patch` format may cause catastrophic forgetting on complex multi-step reasoning tasks. Recommendation: use RL (agentic GRPO) instead of more SFT for capability breadth.

**11. SFT Format Compatibility**

SFT training data (`geak_kernel_sft_v1`) is single-turn, but GEAK uses multi-turn tool-calling. This works because Qwen3-Coder has native tool-calling. SFT adds domain knowledge (kernel optimization) that transfers. The SFT data has 5 task types: `cold_start` (15%), `direction_conditioned` (45%), `error_recovery` (15%), `profile_guided` (15%), `regression_balance` (10%). Error recovery format is useful for multi-turn feedback.

### Benchmark Design

**12. From-Scratch > Optimization**

Optimization benchmarks suffer from measurement noise (identical code reports different speedups) and cheating (autotune deletion, PyTorch replacement). From-scratch writing eliminates these — either the model can write a working kernel or it can't. Target: reach 90% of AITER performance.

**13. Simple vs Complex Operators**

Simple operators (gemm, rms_norm, sampling) are solvable in 50 turns. Complex operators (mha, mla, paged_attention, rope_kv_cache) need 100 turns due to: larger kernels, more complex semantics, multi-kernel coordination.

**14. Prompt Format Per Model**

SFT models should see SFT training format (JSON with contract/direction/baseline). Base model should see generic natural language. Using the wrong format degrades performance. Controlled by `sft_compat` config flag.

## Cheating Prevention (validate_kernel)

Models discover creative ways to bypass Triton kernel requirements. All blocked at `sandbox.write_file` level:

| Pattern | Detection | Example |
|---------|-----------|---------|
| Empty Triton shell + PyTorch wrapper | Check `torch.matmul/mm/bmm/einsum/argsort/sort/gather/cumsum` in wrapper | `@triton.jit def k(): pass` + `def gemm(): return torch.matmul(a,b)` |
| No @triton.jit | Check `@triton.jit` presence | Pure PyTorch implementation |
| Empty kernel body | Check for `tl.load`/`tl.store` in JIT function | `@triton.jit def k(): pass` |
| Method-form PyTorch | Check `.mm(`, `.matmul(`, `.bmm(` | `result = a.matmul(b.T)` |

Blocked ops list: `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.einsum`, `torch.argsort`, `torch.sort`, `torch.gather`, `torch.cumsum`, `torch.nn.functional`, `.mm(`, `.matmul(`, `.bmm(`

## From-Scratch Benchmark Results (SFT-2e)

### Key Findings

**Overall: 9/63 passed (14%), all rms_norm**

| Operator | Tasks | FAST | OK | FAIL | Best Speedup | Notes |
|----------|-------|------|-----|------|-------------|-------|
| rms_norm | 14 | 2 | 7 | 5 | 1.83x | 64% success rate |
| gemm | 19 | 0 | 0 | 19 | 0x (1.66x snapshot) | Can write but unstable |
| mha | 6 | 0 | 0 | 6 | 0x | Too complex |
| mla | 6 | 0 | 0 | 6 | 0x | |
| paged_attn | 6 | 0 | 0 | 6 | 0x | |
| rope_kv | 6 | 0 | 0 | 6 | 0x | |
| sampling | 6 | 0 | 0 | 6 | 0x | |

### Why From-Scratch Fails

1. **SFT data mismatch**: All 4000 SFT samples are input->patch (modify existing kernel). No samples teach from-scratch implementation. The model never saw an empty kernel.py during training.

2. **Triton API gaps**: Model repeatedly hits Triton-specific errors it cannot fix:
   - `tl.dot(a, b.T)` not supported (needs `tl.trans`)
   - Duplicate parameter names in JIT functions
   - Shape mismatches in `tl.dot` reduction dimensions
   - `@triton.autotune` config conflicts with constexpr params

3. **Algorithm complexity**: rms_norm succeeds because it is a 1D reduction (no tiling needed). gemm/mha require 2D tiled computation with shared memory, which the model struggles to implement correctly.

4. **Error loop**: Model often gets stuck in a loop fixing the same error for 200 turns without trying a fundamentally different approach.

### SFT Data Analysis

The existing SFT dataset has 5 task types, ALL with parent_source (existing kernel):
- cold_start (15%): existing kernel, no direction -> patch
- direction_conditioned (45%): existing kernel + direction -> patch
- error_recovery (15%): broken kernel + error -> fix patch
- profile_guided (15%): kernel + profiling data -> optimization patch
- regression_balance (10%): kernel + regression constraints -> balanced patch

**No from-scratch implementation samples exist.** The 14% success rate on from-scratch benchmark reflects Qwen3-Coder base model capability, not SFT training effect.

### Recommendations

1. **Add complete implementation samples** to SFT data: derive from existing 4000 patch samples by applying patches to get final optimized kernels
2. **Add Triton error->fix pairs** from benchmark failure sessions
3. **Keep existing patch data** - it teaches optimization knowledge that transfers
4. **Use agentic RL** for exploration-based learning (from-scratch kernel writing)
5. **SFT evaluations should use optimization mode** (give AITER kernel, model patches it) to match training format

## SFT-f Dataset (Unified Messages Format)

Based on benchmark findings, we augmented the SFT dataset with complete implementation samples and unified all data to `messages` chat format for the next training round (SFT-f).

### Dataset

- HuggingFace: `Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP`
- Local: `/home/danyzhan/geak_sft_dataset/merged_sft_v2/train.jsonl`

### Format

All files unified to `messages` chat format:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert GPU kernel engineer..."},
    {"role": "user", "content": "<task description + kernel code>"},
    {"role": "assistant", "content": "<diff patch or complete kernel>"}
  ],
  "task_type": "...",
  "operator": "...",
  "schema_version": "messages_v1"
}
```

### Composition

| File | Samples | Content |
|------|---------|---------|
| train.jsonl | 3164 | 2000 patch + 1164 complete implementation |
| dev.jsonl | 200 | Patch optimization (converted from geak_kernel_sft_v1) |
| samples.jsonl | 2700 | Full sample pool (converted from geak_kernel_sft_v1) |

### Task Type Distribution (train.jsonl)

| Type | Count | Input | Output | Teaches |
|------|-------|-------|--------|---------|
| direction_conditioned | 907 | kernel + optimization direction | diff patch | Directed optimization |
| complete_implementation | 1164 | operator spec only (no code) | complete kernel | From-scratch writing |
| cold_start | 307 | kernel, no direction | diff patch | Self-directed optimization |
| error_recovery | 293 | broken kernel + error | fix patch | Debugging |
| profile_guided | 293 | kernel + profiling data | optimization patch | Performance tuning |
| regression_balance | 200 | kernel + constraints | balanced patch | Safe optimization |

### Data Sources

- **patch_v1** (2000): From `phase1-production-wave-2000-v1` agent runs
- **complete_impl_v2** (1164): Derived from `phase1-production-wave-2000-v1` by applying patches (fuzzy match) to parent_source to get final optimized kernels. 23 operator types.

### No Overlap with Benchmark

- Training data: `phase1-production-wave-*` task pools (23 operators)
- Benchmark: `held-out-benchmark-aiter` (8 operators, 100 tasks)
- Verified: 0 content overlap

### What Changed from Previous Dataset

1. **Added complete_implementation samples** (1164): Addresses the #1 finding — SFT models couldn't write kernels from scratch because all training data was patch-based. Uses robust fuzzy patch apply to maximize yield from wave-2000.
2. **Full wave-2000 patch data** (2000): All 2000 verified patch samples included (previously only 300 from wave-300)
3. **Unified format**: All data converted from `geak_kernel_sft_v1` structured JSON to `messages` chat format for simpler training pipeline
4. **Covers 23 operator types**: gemm, mha, mla, paged_attention, rms_norm, rope_kv_cache, sampling, and 16 more
