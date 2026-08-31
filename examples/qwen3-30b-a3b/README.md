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

To reproduce the measured MI350X experiments without rebuilding the image,
use the published Docker Hub tag:

```bash
docker pull zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260828

IMAGE_NAME=zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260828 \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

The published image digest is
`sha256:ef5c8632b852f3a04ebc3a9d600e6ef2dc66d4d39e78be41ae97b347ddc6bfd3`.
Set the same `IMAGE_NAME` for benchmark commands that invoke
`run_docker.sh`, or set `IMAGE` when using `run_qwen3_30b_a3b_fsdp.sh`.

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

## Transformers + FSDP2

The FSDP case uses the HuggingFace `Qwen3MoeForCausalLM` implementation with a
Megatron-compatible overlapping dense-DP and EP layout:

- Experts are partitioned across each EP row and tokens are dispatched with
  differentiable all-to-all collectives.
- Every global rank consumes a different microbatch. Shared parameters are
  sharded over the full dense-DP world, while corresponding local experts are
  sharded only over expert-DP replicas (`world_size / EP_SIZE`).
- `--expert-backend` selects `sequential`, `te_grouped`, or `sonic`. The Sonic
  path supports both AITER general-routing and pre-routed APIs and keeps its
  gate/up weights in the interleaved layout required for correct gradients.
- Both BF16 and Lumen FP8 blockwise2d full-parameter training are supported.

Run dense-DP=8, EP=8 on one eight-GPU node:

```bash
NNODES=1 DP_SIZE=8 EP_SIZE=8 \
HOST_MODEL=/path/to/Qwen3-30B-A3B \
HOST_DATA=/path/to/alpaca \
TRAIN_FILE=train.jsonl VAL_FILE=test.jsonl \
bash examples/qwen3-30b-a3b/run_qwen3_30b_a3b_fsdp.sh
```

For two eight-GPU nodes, run the same command on both nodes with `NNODES=2`,
`DP_SIZE=16`, a shared `MASTER_ADDR`, and `NODE_RANK=0`/`1`. Set
`MODE=fp8_blockwise2d` to enable Lumen FP8. The Python entry point can also be
launched directly:

```bash
torchrun --nproc_per_node=8 \
  pretrain_qwen3_30b_a3b_fsdp.py \
  --model-name-or-path /path/to/Qwen3-30B-A3B \
  --train-data-path /path/to/train.jsonl \
  --ep-size 8 --dp-size 8
```

For a from-scratch BF16 accuracy run aligned with the Megatron TE flow
(same architecture, sequence/global batch sizes, optimizer, LR schedule,
router normalization, and normalized auxiliary-loss coefficient), run:

```bash
TRAIN_STEPS=100 COMMAND="bash run_fsdp.sh" \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

To run the GBS=256 comparison at the largest validated microbatch on one
8×MI350X node:

```bash
GBS=256 MBS=4 SEQ_LEN=4096 EXPERT_BACKEND=sonic \
TRAIN_STEPS=20 COMMAND="bash run_fsdp.sh" \
  bash examples/qwen3-30b-a3b/run_docker.sh
```

At sequence length 4096, MBS 32, 16, and 8 exceed the 288 GiB device-memory
limit with activation checkpointing disabled. MBS 4 is the largest divisor of
GBS 256 that completed the one-step memory probe.

`run_docker.sh` defaults to
`zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260828` and overlays the
local FSDP implementation while preserving the image's bundled SonicMoE AITER
sources.
