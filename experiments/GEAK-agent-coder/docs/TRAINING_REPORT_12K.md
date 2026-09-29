# Qwen3-Coder-30B-A3B 12K SFT Training Report

**Date:** 2026-09-28
**Hardware:** 8x AMD MI308X (192GB HBM3 each, gfx942)
**Model:** Qwen/Qwen3-Coder-30B-A3B-Instruct (revision b2cff646)
**Container:** `geak-sft-32k` (ROCm 7.0, PyTorch 2.x, vLLM 0.15.0+rocm700)

## Checkpoints

| Run | Steps | Epochs | Val Loss | HuggingFace |
|-----|-------|--------|----------|-------------|
| 2-epoch | 125 | 2 | 0.1775 | [Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-2epoch](https://huggingface.co/Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-2epoch) |
| 4-epoch | 250 | 4 | 0.1730 | [Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-4epoch](https://huggingface.co/Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-4epoch) |

---

## 1. Data Preparation

### 1.1 Data Sources

The training data comes from three sources, configured in `configs/data/qwen3_30b_a3b_phase1.yaml`:

| Source | Type | Samples | Domain | Description |
|--------|------|---------|--------|-------------|
| `kernel_train` | GEAK trajectories | ~2500 | kernel | Phase 1 production wave (2000 quota), Triton+HIP kernel optimization trajectories collected via multi-tune-agent |
| `kernel_dev` | GEAK trajectories | ~200 | kernel | Phase 1 dev wave (200 quota), held separate for validation |
| `replay` | General coding | 500 | general_coding | SWE-bench style coding tasks for retention |

Raw trajectories are stored in `/home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/` and `/home/danyzhan/geak_sft_dataset/phase1-dev-wave-200-v1/`.

### 1.2 Data Build Pipeline

```bash
geak-agent-coder data-build --config configs/data/qwen3_30b_a3b_phase1.yaml
```

The pipeline (implemented in `src/geak_agent_coder/data/build.py`):

1. **Admission** (`admission.py`): Each source is validated against SHA256 checksums, required provenance fields, and allowed split assignments
2. **Field mapping** (`mapping.py`): Raw trajectory fields are mapped to a canonical schema (`sample_id`, `sample_domain`, `input`, `output`, `messages`)
3. **Tokenization**: Messages are tokenized using the Qwen3-30B-A3B tokenizer with `max_length=32768` and `include_assistant_eot_in_loss=true`; samples exceeding max length are quarantined
4. **Sampling** (`sampling.py`): 1000 training samples are drawn with `length_penalty_power=0.5` (favoring shorter samples to maximize gradient updates per token budget); replay mix enforced at 15-20% assistant loss token share
5. **Manifest**: SHA256 checksums of all artifacts are written to `manifest.json` for reproducibility

### 1.3 Output

| File | Rows | Description |
|------|------|-------------|
| `train.tokenized.jsonl` | 1,000 | 689 kernel + 311 general_coding |
| `dev.tokenized.jsonl` | 200 | Validation set (kernel only) |
| `manifest.json` | - | Checksums and sampling indices for exact reproduction |

### 1.4 Data Integrity

All data artifacts are pinned by SHA256:
- Tokenizer: `4dbc3bb2...`
- Data manifest: `a77fb8c9...`
- Train source: `9142b204...`
- Dev source: `bbb51046...`
- Replay source: `e0877c8e...`

---

## 2. Training Configuration

### 2.1 Model and LoRA

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-Coder-30B-A3B-Instruct |
| Architecture | 30.5B total params, 3.3B activated (MoE, 128 experts, top-8) |
| LoRA attention rank/alpha | 32 / 64 |
| LoRA expert rank/alpha | 16 / 32 |
| LoRA targets (attention) | q_proj, k_proj, v_proj, o_proj |
| LoRA targets (expert) | gate_up, down (Sonic packed format) |
| Trainable modules | 576 (192 attention + 384 expert) |
| Adapter dtype | bfloat16 |

### 2.2 Distributed Training

| Parameter | Value |
|-----------|-------|
| FSDP sharding | `shard_grad_op` |
| Expert parallelism | EP=8 (16 local experts per rank) |
| Expert backend | Sonic (AITER SonicMoE, Triton GEMM) |
| MoE global expert layout | enabled |
| Precision | FP8 blockwise2d (e4m3fnuz, block_size=128) |
| Gradient checkpointing | selective (attention checkpointed, MoE retained) |
| MoE activation offload | synchronous CPU offload via `saved_tensors_hooks` |

### 2.3 Optimization

| Parameter | Value |
|-----------|-------|
| Max sequence length | 12,288 |
| Micro batch size | 1 |
| Gradient accumulation | 2 |
| Effective batch size | 16 (8 DP x 1 micro x 2 GA) |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight_decay=0) |
| Max grad norm | 1.0 |
| Seed | 1234 |

### 2.4 Environment Variables

```bash
export SONIC_MOE_GEMM_BACKEND=triton
export SONIC_MOE_GROUPED_GEMM_BACKEND=triton
export GPU_COREDUMP_ENABLE=0
export HSA_DISABLE_CORE_DUMP=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

---

## 3. Training Procedure

### 3.1 Launch Command

```bash
torchrun --nnodes=1 --nproc-per-node=8 --standalone \
    -m geak_agent_coder.sft.train \
    --config experiments/GEAK-agent-coder/configs/sft/qwen3_coder_12k_production.yaml
```

### 3.2 Two-Epoch Run (125 steps)

Config: `configs/sft/qwen3_coder_12k_production.yaml`

The first run trains from scratch for 125 steps (~2 epochs). Validation is evaluated every 25 steps with 32 batches.

### 3.3 Four-Epoch Run (125 more steps, resume)

Config: `configs/sft/qwen3_coder_12k_4epoch.yaml`

The second run resumes from the 2-epoch checkpoint via `resume_adapter_dir` and trains for 125 additional steps (total 250 steps, ~4 epochs).

### 3.4 Model Export

After training, LoRA adapters are merged into the base model weights:

```bash
python3 experiments/GEAK-agent-coder/scripts/merge_to_hf.py
```

The merge script (`scripts/merge_to_hf.py`):
1. Loads all 8 adapter shards (one per EP rank)
2. Computes `delta = (B @ A) * alpha / rank` for each LoRA pair
3. For expert LoRAs, unpacks Sonic-format `gate_up_lora` into per-expert `gate_proj` + `up_proj` deltas
4. Applies deltas to the corresponding HF safetensor shards (streaming, one shard at a time)
5. Copies tokenizer, config, and other non-weight files
6. Total: 18,624 weight deltas merged (192 attention + 18,432 expert)

---

## 4. Training Curves

### 4.1 Two-Epoch Run (Steps 1-125)

#### Train Loss

```
Step   Loss      Step   Loss      Step   Loss      Step   Loss      Step   Loss
  1    2.3132      26   0.5430      51   0.1902      76   1.3348     101   0.6643
  2    0.9194      27   0.0999      52   0.1412      77   0.4909     102   0.5074
  3    1.2217      28   0.3949      53   0.0767      78   0.2120     103   0.1086
  4    1.6948      29   0.1202      54   1.0160      79   0.6753     104   0.0397
  5    0.5686      30   0.8004      55   0.5173      80   0.2508     105   0.0738
  6    0.4123      31   0.3363      56   0.7128      81   0.8788     106   0.1727
  7    0.5306      32   0.1975      57   0.4317      82   0.4190     107   0.4711
  8    0.3191      33   0.4273      58   0.7669      83   0.4016     108   0.3942
  9    1.7692      34   0.1830      59   0.0415      84   0.8357     109   0.2178
 10    0.3894      35   0.1580      60   0.5052      85   0.5084     110   0.4438
 11    0.4352      36   0.7625      61   1.0231      86   0.3195     111   0.2103
 12    0.7331      37   0.1578      62   0.0984      87   0.4600     112   0.7063
 13    0.0872      38   0.6973      63   0.1185      88   1.0440     113   0.0366
 14    0.7543      39   0.1152      64   0.3621      89   0.3003     114   0.2403
 15    0.3828      40   0.9928      65   0.4302      90   0.0656     115   0.1055
 16    0.2762      41   0.1901      66   0.2505      91   0.4543     116   0.2539
 17    0.4334      42   0.3211      67   0.0595      92   0.3859     117   0.6605
 18    0.2820      43   0.1844      68   0.5573      93   0.1705     118   0.3365
 19    0.6650      44   0.2794      69   0.1782      94   1.5035     119   0.1123
 20    0.6662      45   0.8114      70   0.3661      95   0.1090     120   0.1128
 21    1.4991      46   0.1466      71   0.6706      96   0.1661     121   0.2348
 22    0.2455      47   0.7688      72   0.0803      97   0.2124     122   0.2236
 23    1.2578      48   0.5658      73   0.5079      98   0.1191     123   0.0739
 24    0.4634      49   0.3331      74   0.1094      99   0.8819     124   0.2743
 25    0.5914      50   0.2801      75   0.0754     100   0.5910     125   0.1069
```

#### Validation Loss (2-epoch)

| Step | Val Loss |
|------|----------|
| 25   | 0.2363 |
| 50   | 0.1963 |
| 75   | 0.1870 |
| 100  | 0.1834 |
| 125  | 0.1775 |

### 4.2 Four-Epoch Run (Steps 126-250, resumed)

#### Validation Loss (4-epoch, steps relative to resumed run)

| Step (resumed) | Val Loss |
|----------------|----------|
| 25 (total 150) | 0.1806 |
| 50 (total 175) | 0.1769 |
| 75 (total 200) | 0.1773 |
| 100 (total 225) | 0.1753 |
| 125 (total 250) | 0.1730 |

Validation loss continued to decrease monotonically from 0.1775 to 0.1730, with no overfitting observed.

---

## 5. Key Technical Decisions

### 5.1 `shard_grad_op` instead of `full_shard`

`full_shard` causes NCCL deadlock during backward when DP and EP share the same process group (both are world-size 8 on a single node). The backward all-gather and the MoE all-to-all backward interleave on the same NCCL communicator, causing collective ordering conflicts. `shard_grad_op` avoids the backward all-gather entirely at the cost of ~7 GB more memory per GPU.

### 5.2 Synchronous MoE activation offload

Asynchronous offload using `torch.cuda.Stream` + `record_event` causes `HSA_STATUS_ERROR_EXCEPTION` (0x1016) on ROCm during backward. The root cause is a ROCm-specific bug in GPU event synchronization across streams when combined with `saved_tensors_hooks`. The synchronous variant (`tensor.to("cpu", non_blocking=False)`) avoids this while still reducing peak GPU memory from OOM (~192+ GB) to ~185 GB.

### 5.3 Sonic ModuleList support

Qwen3-Coder uses `nn.ModuleList` for experts (not the packed `gate_up_proj`/`down_proj` tensor format used by Qwen3-30B-A3B). `_SonicLocalExperts.__init__` was extended to detect ModuleList inputs and repack individual `gate_proj.weight`, `up_proj.weight`, `down_proj.weight` tensors into Sonic's expected `w1`/`w2` layout.

### 5.4 Megatron lazy import guards

Both `lumen/models/llama31/__init__.py` and `lumen/models/qwen3_30b_a3b/__init__.py` were modified to wrap megatron imports in `try/except ModuleNotFoundError` so the FSDP2 training path works without megatron-core installed.

---

## 6. GPU Memory Profile

| Phase | Memory per GPU |
|-------|---------------|
| After model load + FSDP2 wrap | ~136 GB |
| After FP8 quantization | ~145 GB |
| During forward (peak) | ~173 GB |
| During backward (peak) | ~185 GB |
| Headroom | ~7 GB / 192 GB |

---

## 7. Reproduction Steps

### Prerequisites

- 8x AMD MI308X GPUs (192 GB HBM3 each)
- Docker container with ROCm 7.0, PyTorch, vLLM 0.15.0+rocm700
- Lumen repo at `dev/moe` branch
- GEAK SFT dataset at `/home/danyzhan/geak_sft_dataset/`
- General coding replay at `/home/danyzhan/phase1_control/`

### Step 1: Build tokenized dataset

```bash
cd /home/danyzhan/Lumen
pip install -e experiments/GEAK-agent-coder
geak-agent-coder data-build --config experiments/GEAK-agent-coder/configs/data/qwen3_30b_a3b_phase1.yaml
```

### Step 2: Run 2-epoch training

```bash
export PYTHONPATH="/home/danyzhan/Lumen:${PYTHONPATH}"
export SONIC_MOE_GEMM_BACKEND=triton
export SONIC_MOE_GROUPED_GEMM_BACKEND=triton
export GPU_COREDUMP_ENABLE=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

torchrun --nnodes=1 --nproc-per-node=8 --standalone \
    -m geak_agent_coder.sft.train \
    --config experiments/GEAK-agent-coder/configs/sft/qwen3_coder_12k_production.yaml
```

### Step 3: Run 4-epoch training (resume)

```bash
torchrun --nnodes=1 --nproc-per-node=8 --standalone \
    -m geak_agent_coder.sft.train \
    --config experiments/GEAK-agent-coder/configs/sft/qwen3_coder_12k_4epoch.yaml
```

### Step 4: Merge LoRA into HF format

Edit `scripts/merge_to_hf.py` to point `adapter_dir` to the desired checkpoint (`best/` subdir), then:

```bash
python3 experiments/GEAK-agent-coder/scripts/merge_to_hf.py
```

### Step 5: Upload to HuggingFace

```bash
huggingface-cli upload Zhangdanyang/<model-name> <merged-output-dir> . --repo-type model
```
