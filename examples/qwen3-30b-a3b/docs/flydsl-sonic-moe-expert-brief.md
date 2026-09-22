# Qwen3-30B-A3B FlyDSL SonicMoE — operator-level notes

I can only run the full model on 8 GPUs. On your side a single-GPU kernel bench is enough; below is what **one expert call** looks like in training. End-to-end step time (~12 s) is dominated by EP all-to-all and attn CUDA graphs — not the optimization target.

---

## What I measured on the full model

8×MI350X, Megatron BF16, seq=4096, MBS=2, GBS=256, EP=8, TE `--cuda-graph-scope attn`. One step = 48 layers × 16 microbatches = **768 expert calls**. GPU time is `SonicMoE.experts.fwd` / `.bwd` on a profiled graph-replay step (that step’s wall is inflated). Unprofiled neighbors are the e2e wall.

**Pin used for the kernel-split table below:** FlyDSL `98764e7` vs AITER `8b56778` (2026-09-19).

| | AITER | FlyDSL | vs AITER |
|---|---|---|---|
| expert **fwd** | 1.32 ms/call · **1.01 s/step** | 1.30 ms/call · **0.995 s/step** | parity |
| expert **bwd** | 4.86 ms/call · **3.73 s/step** | 3.78 ms/call · **2.90 s/step** | **−22%** |
| experts total | 4.74 s/step | 3.90 s/step | −18% |
| share of e2e wall | **39%** | **33%** | |

| | AITER + TE graph | FlyDSL + TE graph |
|---|---|---|
| step 5 | 12.17 s | 11.92 s |
| step 7 | 11.92 s | 11.93 s |
| e2e wall (mean) | **12.05 s** | **11.93 s** |
| `mem usages` | 0.679 | 0.815 |

**Newer logs (do not mix into the kernel table):** FlyDSL `0cb93ba` e2e vs AITER `ccd9200b` TE-graph, same recipe. Unprofiled step 5/7: AITER **12.72 / 12.70 s** (`mem 0.694`), FlyDSL **11.07 / 11.05 s** (`mem 0.750`). FlyDSL steps 11–20 median **11.02 s**. Profiler CUDA on that FlyDSL pin: experts.fwd **~1.28 ms/call**, experts.bwd **~3.11 ms/call**.

Lumen pre-routed microbench (same ABI as training): `tests/kernels/bench_sonic_moe_aiter_vs_flydsl.py`.

E2E is tied because EP `all_to_all` still burns ~6 s of GPU time (overlapped). Expert bwd is already 0.83 s/step faster than AITER and that only moves the wall by ~0.1 s. Please keep reporting **ms/call fwd and bwd**, not e2e step time. A single-GPU bench will not match 1.30 / 3.78 ms absolutely (no A2A/graph on the same streams); the **ratio vs AITER** and the kernel split should.

---

## One call (what to bench on one GPU)

After Megatron router / permute / EP all-to-all, the kernel sees **expert-major BF16 rows**:

- `E=16, H=2048, I=768` (SwiGLU; gemm1 N = `2I=1536`)
- Mean `T = MBS × seq × topk / EP = 2 × 4096 × 8 / 8 = 8192` routes
- `token_indices = arange(T)`, `cu_seqlens` from 16 expert counts

```python
forward_routes_training(
    hidden, token_indices, expert_indices, scores,
    expert_offsets=cu_seqlens,
    token_indices_identity=True,
)
# bwd: sonic_moe_backward_routes(..., forward_state=retained, token_indices_sorted=True)
```

Do not use `SonicMoE(x, router_logits)`, and do not re-run the generic ragged counting sort. `T` changes every training step — do not gate the fast path on `routes==8192`. Baseline AITER on the same pre-routed tensors (`moe_pre_routed_inputs` / `SonicMoEExperts` with `SONIC_MOE_GEMM_BACKEND=triton`), not a token-major path that still sorts.

---

## Cases I would sweep on one GPU

`hidden [T,2048] bf16`, `counts [16] int32` → `cu_seqlens`, `scores [T] fp32`. Warmup 5, median of 20. Split fwd and bwd.

**T**

| T | Why |
|---|---|
| 8192 | Balanced mean; old exact fast path |
| 8191 | Regression: T≠8192 used to fall back to generic sort |
| 4096 / 16384 / 32768 | Typical EP imbalance |
| 65536 | Hot expert / large T I have seen |

**counts** (every T): balanced (`≈T/16` each), skew (softmax multinomial), hot4 (~80% on 4 experts).

**Workspace**: same process, `8192 → 65536 → 4096 → 16384`. Watch whether workspace only grows and peak GB. In e2e FlyDSL reserved ~0.82 vs AITER ~0.68; graph capture is tight.

---

## Kernel split from my e2e step (÷768 → per call)

FlyDSL-owned:

| Kernel | Per call |
|---|---|
| `fused_prepare_kernel_0` | **1.08 ms** (≈ full fwd) |
| `gemm1_a16w4_port` | 0.69 ms |
| `gemm2_a16w4` | 0.36 ms |
| `sonic_grouped_tn` m1536×n2048 | 0.71 ms |
| `sonic_grouped_nn` k1536×n2048 | 0.55 ms |
| `sonic_grouped_tn` m2048×n768 | 0.51 ms |
| `sonic_grouped_nn` k2048×n768 | 0.26 ms |
| `fused_kernel_0` | 0.28 ms |
| `_FlyDSLNativeRoutesBackward` | 3.52 ms |

AITER, same step: `_grouped_gemm_kernel` is more fragmented; dw ~0.59 ms/call; Up/Down projection bwd ~0.5–1.5 ms. Your GEMMs are already fine.

Please look at:

1. Skipping or fusing `fused_prepare_kernel_0` when retained-state + `expert_offsets` are present.
2. Fewer launches for the four `sonic_grouped_*` backward GEMMs.
3. Reusing workspace across changing `T` instead of pinning one buffer per T.

Megatron `_moe_chunk_sort` / `_unpermute` are not yours.

---

## What “done” looks like

One table: `T, dist, flydsl_fwd, flydsl_bwd, aiter_fwd, aiter_bwd, peak_mem`. Success is **lower bwd ms** (especially T≠8192 and skew) and **peak mem that does not grow linearly with the number of distinct T**, with fwd staying at AITER parity.
