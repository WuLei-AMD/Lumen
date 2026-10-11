# MI308 Implementation and Execution Runbook

## Current phase: implementation only

Allowed before the final replay dataset is frozen:

- CPU unit tests and fake/tiny-model tests;
- CLI/config validation;
- dataset gate dry-runs that are expected to fail closed;
- generation of design manifests without training artifacts.

Not allowed yet:

- 8-GPU SFT;
- full 30B adapter merge or FP8 export;
- model serving;
- GEAK GPU pre/post benchmark.

## Production prerequisites

1. General Coding Replay has exactly 500 accepted rows.
2. Leakage, trust, quota-v2, token-mix and checksum reports pass.
3. Kernel Train 2,000 and Kernel Dev 200 manifests are pinned.
4. The `Qwen/Qwen3-30B-A3B` tokenizer revision and chat-template hash are
   pinned.
5. Export length sweep chooses a no-truncation sequence length or an approved
   quarantine policy.
6. Base model and output storage have immutable identities and sufficient
   capacity.

## Intended training topology

```text
node:                    1
MI308X GPUs:             8
architecture:            gfx942
world size / DP size:    8
expert parallel size:    8
expert backend:          sequential
FSDP strategy:           full_shard initially
micro batch:             1 initially
gradient checkpointing:  enabled initially
FP8:                     E4M3 blockwise2d, 128×128
LoRA:                    attention r32 + experts r8
```

The first authorized GPU action is a short finite-loss smoke run, not the
production SFT. It must verify adapter coverage, frozen base gradients,
checkpoint save/resume and BF16-vs-FP8 loss behavior before scaling duration.

The shipped SFT config remains disabled with `execution.allow_training: false`.
Authorization additionally requires an immutable HF revision, replay package
root, tokenizer SHA256, and final data-manifest path/SHA256. The launcher
validates the data artifact and replay finalization reports before initializing
the distributed runtime.
The configured Dev split is evaluated periodically for best-adapter selection
and bounded early stopping.

## Promotion sequence

```text
data preflight
→ tokenizer export and length report
→ tiny/CPU tests
→ 8-GPU 5-step smoke
→ save/resume smoke
→ bounded SFT
→ Dev checkpoint selection
→ adapter merge
→ candidate HF FP8 export
→ vLLM smoke
→ fixed-agent Base/SFT benchmark
→ retention and paired-CI gates
```

Every stage writes a manifest and consumes hashes from the previous stage.
Failure never promotes a partial artifact.

## Reusable AMD profiles

MI308-specific values now live in
`configs/hardware/amd_mi308_gfx942.yaml` and the immutable
`geak_mi308_release_v1` recipe. MI300/gfx942, MI350/gfx950 and generic
ROCm-BF16 profiles are separate files. Switching profile changes topology and
precision capability checks; it does not weaken the GEAK release recipe.

The launcher probes `gcnArchName` before RCCL initialization. A profile/runtime
mismatch or an FP8 request on a profile without native FP8 support is fatal.
There is no silent BF16 fallback.

## SFT-f Training (2026-10-11)

### Dataset: 3164 samples in unified messages format

SFT-f uses the expanded dataset with both patch optimization and from-scratch
kernel writing samples. All data in `messages` chat format.

| Component | Count | Source |
|-----------|-------|--------|
| Patch optimization | 2000 | `phase1-production-wave-2000-v1` |
| Complete implementation | 1164 | Derived via fuzzy patch apply |
| Dev | 200 | `phase1-dev-wave-200-v1` |

HuggingFace: `Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP`

### Data pipeline

```bash
cd /home/danyzhan/Lumen/experiments/GEAK-agent-coder

# 1. Generate complete implementation samples from wave-2000
python3 scripts/gen_complete_impl.py

# 2. Unify all SFT data to messages format
python3 scripts/unify_sft_format.py

# 3. Tokenize into input_ids + loss_mask
python3 scripts/tokenize_messages_dataset.py
# Output: data/build/qwen3_30b_a3b_sft_f/{train,dev}.tokenized.jsonl

# 4. Upload to HuggingFace
huggingface-cli upload Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP \
    /home/danyzhan/geak_sft_dataset/merged_sft_v2/train.jsonl train.jsonl \
    --repo-type dataset
```

### Training commands

```bash
cd /home/danyzhan/Lumen/experiments/GEAK-agent-coder
export PYTHONPATH="src:${PYTHONPATH:-}"

# 2-epoch (198 steps)
python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --standalone \
    -m geak_agent_coder.sft.train \
    --config configs/sft/qwen3_coder_sft_f_2epoch.yaml

# Merge 2-epoch adapter
python3 scripts/merge_to_hf.py \
    --adapter-dir outputs/qwen3-coder-sft-f-2epoch \
    --base-model models/Qwen3-Coder-30B-A3B-Instruct \
    --output-dir outputs/qwen3-coder-sft-f-2epoch-merged

# 4-epoch (resume from 2-epoch, 198 more steps)
python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --standalone \
    -m geak_agent_coder.sft.train \
    --config configs/sft/qwen3_coder_sft_f_4epoch.yaml

# Merge 4-epoch adapter
python3 scripts/merge_to_hf.py \
    --adapter-dir outputs/qwen3-coder-sft-f-4epoch \
    --base-model models/Qwen3-Coder-30B-A3B-Instruct \
    --output-dir outputs/qwen3-coder-sft-f-4epoch-merged
```

### Training parameters

```text
samples:                 3164 (2000 patch + 1164 complete_impl)
max_sequence_length:     12288
batch size:              8 GPUs × 1 micro_bs × 4 GA = 32
steps/epoch:             ~99
2-epoch steps:           198
4-epoch steps:           198 (resume from 2e)
evaluation_interval:     50 steps
expert_backend:          sonic
FP8:                     E4M3 blockwise2d
LoRA:                    attention r32 + experts r8
```

### Configs

| File | Purpose |
|------|---------|
| `configs/sft/qwen3_coder_sft_f_2epoch.yaml` | 2-epoch training |
| `configs/sft/qwen3_coder_sft_f_4epoch.yaml` | 4-epoch (resume from 2e) |
| `data/build/qwen3_30b_a3b_sft_f/manifest.json` | Tokenized data manifest |

### Outputs

| Artifact | Path |
|----------|------|
| 2-epoch adapter | `outputs/qwen3-coder-sft-f-2epoch/` |
| 2-epoch merged | `outputs/qwen3-coder-sft-f-2epoch-merged/` |
| 4-epoch adapter | `outputs/qwen3-coder-sft-f-4epoch/` |
| 4-epoch merged | `outputs/qwen3-coder-sft-f-4epoch-merged/` |
| Training log | `/home/danyzhan/sft_f_training.log` |

### Key differences from previous SFT runs

1. **3164 vs 2000 samples**: Added 1164 complete_implementation samples
2. **Messages format**: Direct chat format instead of geak_kernel_sft_v1
3. **Custom tokenizer**: Uses `tokenize_messages_dataset.py` instead of `data-build`
4. **Loss on assistant+EOT**: Full assistant content gets loss=1
