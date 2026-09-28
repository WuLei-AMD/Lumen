# GEAK Agent Coder

Config-driven tooling for LoRA SFT of `Qwen/Qwen3-30B-A3B` on AMD MI-series
ROCm systems and for paired pre/post-training GEAK evaluation. MI308X/gfx942
remains the pinned production profile, not a Python-level runtime assumption.

This directory contains four fail-closed stages:

1. validate and export the frozen Hugging Face Phase 1 dataset;
2. train BF16 LoRA adapters over a frozen Lumen FP8 base model;
3. merge adapters and create a candidate HF blockwise-FP8 checkpoint;
4. compare the base and tuned endpoints with an identical GEAK agent loop.

## Current execution state

The implementation can be developed and tested before the dataset is ready,
but training is intentionally gated. A production run is forbidden until the
General Coding Replay finalization reports all of the following:

- exactly 500 accepted rows;
- leakage status `pass`;
- package status `ready`;
- frozen checksums and a valid 15–20% assistant-loss-token mix.

At the time this project was created, 500 receipts had passed offline CPU
double replay, but the final pool still had 114 Dev repository overlaps and
only 379 deduplicated rows. The 209-row MEnvData wave-2 candidate pool was
frozen but had not been verified.

No training, 30B export, model serving, or GPU benchmark is performed by the
test suite.

## Training architecture

- Backend: Transformers + Lumen FSDP2, `EP=8`.
- Base checkpoint: `Qwen/Qwen3-30B-A3B`.
- Expert backend: `sequential`, because its local experts expose patchable
  `nn.Linear` modules.
- Attention LoRA: `q_proj`, `k_proj`, `v_proj`, `o_proj`, rank 32, alpha 64.
- Expert LoRA: `gate_up_proj`, `down_proj`, rank 8, alpha 16.
- Router, embeddings, norms, and `lm_head`: frozen.
- Base linear compute: E4M3 `blockwise2d`, 128×128 blocks.
- LoRA parameters and optimizer path: BF16.

SonicMoE is not treated as expert-LoRA compatible. Its fused `w1`/`w2`
implementation does not expose the gate/up activation boundary required for
an equivalent standard LoRA update.

## Safety boundary

Training FP8 and deployment FP8 are separate artifacts. Training retains a
master base checkpoint plus BF16 adapters. Deployment first merges the
adapters, then emits a candidate HF FP8 checkpoint. A candidate is not marked
deployable until tensor coverage, metadata, numerical checks, and a real ROCm
vLLM load/generate smoke test have passed.

See `docs/` for the data, LoRA, export, benchmark, and MI308 run contracts.

## Reuse with another dataset or AMD GPU

Configuration is composed from three independent layers:

- `configs/hardware/`: GPU architecture, topology and precision capabilities;
- `configs/recipes/`: model, LoRA, precision and optimization policy;
- `configs/data/`: raw source mappings, admission gates, tokenization and
  sampling.

Build a local, immutable training artifact first:

```bash
geak-agent-coder data-build --config configs/data/generic_jsonl.yaml
```

Then select a training composition such as
`configs/sft/mi300_jsonl_fp8.yaml` or `configs/sft/mi350_hf_fp8.yaml`. Pin the
model revision, tokenizer SHA256, data manifest path/SHA256 and set
`execution.allow_training: true` only after reviewing the generated manifest.
The training launcher checks the live ROCm architecture before initializing
distributed state.

Raw inputs can be local JSONL, a local Hugging Face layout, or a pinned dataset
already present in the HF cache. Network access is disabled by default.
Declarative mappings normalize fields into `generic_coding_sft_v1`,
`geak_kernel_sft_v1`, or the strictly gated
`general_coding_replay_v1`. Sampling is materialized into the emitted training
JSONL, so the configured schedule is the schedule consumed by the trainer.

This reuse boundary keeps the Qwen3-30B-A3B Lumen backend. A different model
architecture still needs its own expert-sharding and LoRA target backend; it
cannot safely be enabled by changing YAML alone.
