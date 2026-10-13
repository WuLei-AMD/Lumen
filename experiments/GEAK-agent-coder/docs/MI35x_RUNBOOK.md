# MI35x Qwen3-Coder SFT Runbook

## 1. Scope

This runbook reproduces the GEAK kernel-coder SFT run performed on one node
with 8 AMD Instinct MI350X GPUs (`gfx950`).

- Base model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
- Base revision: `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`
- Dataset: `Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP`
- Dataset revision: `82228c7688de3238f40008aa5cfaa3a8e99f82ee`
- Train/dev rows: 3,164 / 200
- Docker image: `lumen/mxfp4-atom:gfx950`
- Topology: EP=8, DP/world size=8
- Training precision: BF16 master weights and adapters, with Lumen E4M3
  blockwise2d FP8 compute on attention base linear layers
- Export precision: E4M3 (`torch.float8_e4m3fn`) with 128x128 weight blocks

The inference FP8 checkpoint is not used as a trainable master checkpoint.
Training starts from the BF16 model and applies FP8 compute through Lumen.

## 2. Checkout

```bash
cd /home/danyzhan

git clone --branch dev/moe --single-branch \
  https://github.com/ZhangDanyang-AMD/Lumen.git

git clone --branch dev/multi-loop-agent --single-branch \
  https://github.com/ZhangDanyang-AMD/Lumen-RL.git

cd /home/danyzhan/Lumen
git submodule update --init --recursive third_party/aiter
```

The pinned AITER checkout must contain
`aiter/ops/triton/sonicmoe.py`. Initializing its nested
`composable_kernel` submodule is required for JIT compilation.

## 3. Download immutable inputs

Use a Hugging Face read token that has accepted the gated dataset terms.
Do not commit the token.

```bash
export HF_TOKEN="<read-token>"

docker run --rm \
  -e HF_TOKEN \
  -v /home/danyzhan:/home/danyzhan \
  --entrypoint bash \
  lumen/mxfp4-atom:gfx950 -lc '
    hf download \
      Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP \
      train.jsonl dev.jsonl \
      --repo-type dataset \
      --revision 82228c7688de3238f40008aa5cfaa3a8e99f82ee \
      --local-dir /home/danyzhan/geak_sft_dataset/hf-main

    hf download \
      Qwen/Qwen3-Coder-30B-A3B-Instruct \
      --revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
      --local-dir \
      /home/danyzhan/Lumen/experiments/GEAK-agent-coder/models/Qwen3-Coder-30B-A3B-Instruct
  '
```

Expected row counts:

```text
train.jsonl  3164
dev.jsonl     200
```

## 4. Tokenize without truncation

The tokenizer script scans train and dev before writing any output. By default,
it rounds the observed maximum up to a multiple of 8 and fails if any row would
overflow.

```bash
docker run --rm \
  -v /home/danyzhan:/home/danyzhan \
  --entrypoint bash \
  lumen/mxfp4-atom:gfx950 -lc '
    cd /home/danyzhan/Lumen/experiments/GEAK-agent-coder
    python3 scripts/tokenize_messages_dataset.py \
      --train /home/danyzhan/geak_sft_dataset/hf-main/train.jsonl \
      --dev /home/danyzhan/geak_sft_dataset/hf-main/dev.jsonl \
      --output-dir \
        /home/danyzhan/Lumen/experiments/GEAK-agent-coder/data/build/qwen3_30b_a3b_sft_f_mi350
  '
```

Observed values for the pinned dataset:

```text
train maximum:          10,868 tokens
dev maximum:             5,152 tokens
configured maximum:     10,872 tokens
overflow/quarantine:         0 rows
```

The generated `manifest.json` has schema
`geak_training_data_manifest_v2` and pins the tokenizer, sources, and tokenized
artifacts by SHA256.

## 5. Validate configuration

Training configs:

- `configs/sft/qwen3_coder_sft_f_mi350_smoke.yaml`
- `configs/sft/qwen3_coder_sft_f_mi350_2epoch.yaml`
- `configs/sft/qwen3_coder_sft_f_mi350_4epoch.yaml`

The 4-epoch config resumes from the completed 2-epoch adapter and performs 198
additional optimizer steps.

```bash
docker run --rm \
  -v /home/danyzhan:/home/danyzhan \
  --entrypoint bash \
  lumen/mxfp4-atom:gfx950 -lc '
    cd /home/danyzhan/Lumen/experiments/GEAK-agent-coder
    PYTHONPATH=src python3 - <<'"'"'PY'"'"'
from geak_agent_coder.sft.launcher import load_config, assert_training_authorized

for path in (
    "configs/sft/qwen3_coder_sft_f_mi350_smoke.yaml",
    "configs/sft/qwen3_coder_sft_f_mi350_2epoch.yaml",
    "configs/sft/qwen3_coder_sft_f_mi350_4epoch.yaml",
):
    config = load_config(path)
    assert_training_authorized(config)
    print(path, config["hardware"]["target_arch"])
PY
  '
```

## 6. Docker launch

Common Docker options:

```bash
IMAGE=lumen/mxfp4-atom:gfx950
PYTHONPATH_IN_CONTAINER=/home/danyzhan/Lumen/third_party/aiter:/home/danyzhan/Lumen:/home/danyzhan/Lumen/experiments/GEAK-agent-coder/src
```

### 6.1 One-step smoke

Always run the finite smoke before production.

```bash
docker run --rm \
  --name geak-sft-mi350-smoke \
  --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size=64g \
  --ulimit memlock=-1:-1 \
  -v /home/danyzhan:/home/danyzhan \
  -e PYTHONPATH="${PYTHONPATH_IN_CONTAINER}" \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e GPU_ARCHS=gfx950 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TORCH_EXTENSIONS_DIR=/home/danyzhan/.cache/torch_extensions_gfx950 \
  --entrypoint bash \
  "${IMAGE}" -lc '
    cd /home/danyzhan/Lumen/experiments/GEAK-agent-coder
    python3 -m torch.distributed.run \
      --nnodes=1 --nproc-per-node=8 --standalone \
      -m geak_agent_coder.sft.train \
      --config configs/sft/qwen3_coder_sft_f_mi350_smoke.yaml
  '
```

The completed smoke validated:

- all 8 ranks detect `gfx950`;
- AITER SonicMoE imports and JIT builds;
- 192 attention linear modules use E4M3 blockwise2d FP8;
- finite forward/backward loss;
- adapter checkpoint save.

### 6.2 Two epochs, then resume to four epochs

```bash
docker run --rm \
  --name geak-sft-mi350 \
  --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size=64g \
  --ulimit memlock=-1:-1 \
  -v /home/danyzhan:/home/danyzhan \
  -e PYTHONPATH="${PYTHONPATH_IN_CONTAINER}" \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e GPU_ARCHS=gfx950 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TORCH_EXTENSIONS_DIR=/home/danyzhan/.cache/torch_extensions_gfx950 \
  --entrypoint bash \
  "${IMAGE}" -lc '
    set -o pipefail
    cd /home/danyzhan/Lumen/experiments/GEAK-agent-coder

    python3 -m torch.distributed.run \
      --nnodes=1 --nproc-per-node=8 --standalone \
      -m geak_agent_coder.sft.train \
      --config configs/sft/qwen3_coder_sft_f_mi350_2epoch.yaml \
      2>&1 | tee /home/danyzhan/sft_f_mi350_2epoch.log

    python3 -m torch.distributed.run \
      --nnodes=1 --nproc-per-node=8 --standalone \
      -m geak_agent_coder.sft.train \
      --config configs/sft/qwen3_coder_sft_f_mi350_4epoch.yaml \
      2>&1 | tee /home/danyzhan/sft_f_mi350_4epoch.log
  '
```

Expected adapter outputs:

```text
outputs/qwen3-coder-sft-f-mi350-2epoch/
outputs/qwen3-coder-sft-f-mi350-4epoch/
```

Each output contains one shared/rank-0 shard and seven EP-local expert shards.

## 7. Training observations

Completed loss summary:

```text
             first-10 train   last-10 train   validation
2 epoch           0.4362          0.1598      0.1790 → 0.1673 → 0.1620
4 epoch resume    0.1705          0.1284      0.1567 → 0.1553 → 0.1568
```

All logged losses were finite. Long samples can temporarily drive HBM close to
capacity and produce allocator warnings before cache recovery. Treat a warning
as fatal only if a rank exits, a traceback follows, or progress stops.

### Current Sonic expert-LoRA limitation

The current `PackedExpertLoRA` modules are registered and checkpointed but are
not consumed by the Sonic expert forward path. Their `lora_B` tensors therefore
remain zero. The completed run effectively trained attention LoRA only.

This does not corrupt the checkpoint, but the run must not be described as
expert-LoRA training. Wire packed expert adapters into Sonic forward and repeat
the smoke before claiming expert updates.

## 8. Merge adapters

```bash
cd /home/danyzhan/Lumen/experiments/GEAK-agent-coder

python3 scripts/merge_to_hf.py \
  --base-model models/Qwen3-Coder-30B-A3B-Instruct \
  --adapter-dir outputs/qwen3-coder-sft-f-mi350-2epoch \
  --output-dir outputs/qwen3-coder-sft-f-mi350-2epoch-merged

python3 scripts/merge_to_hf.py \
  --base-model models/Qwen3-Coder-30B-A3B-Instruct \
  --adapter-dir outputs/qwen3-coder-sft-f-mi350-4epoch \
  --output-dir outputs/qwen3-coder-sft-f-mi350-4epoch-merged
```

The completed merge audit showed:

```text
unchanged base tensors, bit-identical: 18,675
changed attention tensors:                192
non-finite changed tensors:                 0
```

Run a direct Transformers `generate()` smoke on each merged BF16 checkpoint
before FP8 export. Both completed merged checkpoints generated valid code and
kernel diffs.

## 9. Export fixed FP8 checkpoints

```bash
export PYTHONPATH=/home/danyzhan/Lumen/experiments/GEAK-agent-coder/src
cd /home/danyzhan/Lumen/experiments/GEAK-agent-coder

python3 scripts/export_sft_fp8.py \
  --source-dir outputs/qwen3-coder-sft-f-mi350-2epoch-merged \
  --output-dir outputs/qwen3-coder-sft-f-mi350-2epoch-fp8-fixed \
  --target-arch gfx950 \
  --fp8-format e4m3fn \
  --block-size 128

python3 scripts/export_sft_fp8.py \
  --source-dir outputs/qwen3-coder-sft-f-mi350-4epoch-merged \
  --output-dir outputs/qwen3-coder-sft-f-mi350-4epoch-fp8-fixed \
  --target-arch gfx950 \
  --fp8-format e4m3fn \
  --block-size 128
```

The export uses portable HF/vLLM metadata:

```json
{
  "quant_method": "fp8",
  "fmt": "e4m3",
  "activation_scheme": "dynamic",
  "weight_block_size": [128, 128],
  "modules_to_not_convert": ["..."]
}
```

`modules_to_not_convert` is generated from preserved weights. It prevents
routers, norms, embeddings, and `lm_head` from being replaced by FP8 modules
without matching scale tensors.

Exported tensor coverage:

```text
quantized:
  attention q/k/v/o projections                 192
  48 layers x 128 experts x gate/up/down     18,432
  total                                      18,624

preserved:
  routers, Q/K norms, layer norms,
  embeddings, final norm, lm_head               243
```

Fixed artifact identities:

```text
2 epoch: 977915ae1ac30f40bf089027719b64dd059c966d950d2f5c982033879abd64d7
4 epoch: 37c7d734c84f05b7cd384b7e99510017b4ce5ce23ea24a6f9c52205268f69768
```

## 10. Validation gates

Before publishing:

1. Confirm `fmt == "e4m3"`.
2. Confirm all 48 routers and `lm_head` are in `modules_to_not_convert`.
3. Confirm 18,624 quantized weights and 16 safetensor shards.
4. Confirm the manifest artifact SHA256.
5. Load with Transformers using `FineGrainedFP8Config(dequantize=True)` and run
   `generate()`.
6. Run a native FP8 load/generate smoke in the exact benchmark vLLM image.
7. Only call `mark_deployable()` after successful native smoke evidence.

The generic Transformers Triton FP8 kernel in the training image aborts on
ROCm/gfx950 during native execution. This is a runtime-kernel limitation of that
image, not evidence of bad checkpoint contents. Dequantized Transformers
generation succeeded for both fixed exports.

## 11. Publish to Hugging Face

Use a write token through the environment. Never put it in the command history,
runbook, model directory, or Git repository.

```bash
export HF_TOKEN="<write-token>"

hf upload-large-folder \
  Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-2epoch \
  outputs/qwen3-coder-sft-f-mi350-2epoch-fp8-fixed \
  --repo-type model \
  --revision main \
  --exclude README.md \
  --num-workers 8

hf upload-large-folder \
  Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-4epoch \
  outputs/qwen3-coder-sft-f-mi350-4epoch-fp8-fixed \
  --repo-type model \
  --revision main \
  --exclude README.md \
  --num-workers 8
```

Published revisions at the time of this run:

```text
2 epoch: 11b490c70a7419f1b87e86819580e213d76baf8a
4 epoch: 4e31dc71c6ca5fffcbe26e505a3a5e59242bd10a
```

Verify the remote `config.json`, `fp8_manifest.json`, shard count, and artifact
SHA256 after upload.

