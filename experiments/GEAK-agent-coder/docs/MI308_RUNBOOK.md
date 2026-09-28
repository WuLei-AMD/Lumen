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
