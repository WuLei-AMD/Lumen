# Qwen3-30B-A3B Megatron + SonicMoE

This example runs the Qwen3-30B-A3B architecture with Megatron on one
8×MI350X node. It supports three interchangeable expert implementations:

- `sequential`: Megatron `SequentialMLP`
- `te_grouped`: Transformer Engine `TEGroupedMLP`
- `sonic`: AITER pure-Triton SonicMoE, installed through
  `LumenConfig.enable(model)`

## Build

```bash
git submodule update --init third_party/aiter
bash examples/qwen3-30b-a3b/build.sh
```

The image is built from the official ROCm 7.2 PyTorch base. Transformer Engine
`v2.10_rocm`, ROCm Megatron-LM, the SonicMoE AITER revision, and Lumen are all
built from source; it does not inherit Miles or Primus.

The Lumen AITER submodule must point to the SonicMoE-enabled
`lumen/qwen3-30b-a3b` revision.

## SequentialMLP versus TEGroupedMLP

```bash
COMMAND="bash benchmark_mlp.sh" \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

The default benchmark runs 20 iterations at sequence length 4096 and writes
per-run logs plus `results/mlp_summary.csv`. Override `TRAIN_STEPS`, `SEQ_LEN`,
`MBS`, and `GBS` through the environment.

## SonicMoE e2e training

First run a short smoke:

```bash
MOE_IMPL=sonic TRAIN_STEPS=5 SEQ_LEN=1024 \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

Then run the same workload as the baselines:

```bash
MOE_IMPL=sonic TRAIN_STEPS=20 SEQ_LEN=4096 \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

## Kernel-level Qwen shape

With EP=8, each GPU owns 16 experts. Top-8 routing produces roughly eight
dispatched rows per source token after all-to-all, so the representative local
SonicMoE shape at sequence length 4096 is:

```bash
cd /workspace/Lumen/third_party/aiter
python op_tests/test_sonicmoe.py \
  --activation swiglu --benchmark \
  --T 32768 --H 2048 --I 768 --E 16 --K 1
```

SonicMoE tuning data is stored under
`third_party/aiter/aiter/ops/triton/configs/moe/`. Tune against the kernel
benchmark first, then verify gains with the e2e `sonic` run because routing,
all-to-all, and weight-gradient costs are not represented by GEMM-only timing.

## MI350X reference results

The following BF16 results were measured on one 8×MI350X node with TP=1,
EP=8, MBS=1, GBS=8, and gradient-accumulation fusion disabled:

- Sequence 256: SequentialMLP 608 ms/step; TEGroupedMLP 616 ms/step.
- Sequence 4096: SequentialMLP 1198 ms/step; TEGroupedMLP 2109 ms/step.
- SonicMoE kernel at T=32768: forward 0.77 ms, backward 3.70 ms,
  4.46 ms total (277 TFLOPS).
- SonicMoE e2e at sequence 4096 (steps 4–12): 2614 ms median,
  1641 ms best. The expert-major storage fix improved the previous
  post-JIT median by about 2.6×, but the result remains variable and slower
  than SequentialMLP.

SonicMoE tuning removes host synchronization from grouped-GEMM grid
calculation and preserves expert-major physical weight layout. E2E profiling
should focus next on integration/optimizer scheduling rather than GEMM tiles.
