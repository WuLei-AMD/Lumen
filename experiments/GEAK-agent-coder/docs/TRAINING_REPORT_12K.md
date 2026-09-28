# Qwen3-Coder-30B-A3B 12K SFT Training Report

**Date:** 2026-09-28
**Hardware:** 8x AMD MI308X (192GB HBM3 each)
**Model:** Qwen/Qwen3-Coder-30B-A3B-Instruct (revision b2cff646)
**HF Upload:** [Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-2epoch](https://huggingface.co/Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-2epoch)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Max sequence length | 12,288 |
| Total steps | 125 |
| Epochs | ~2 |
| Gradient accumulation | 2 |
| Micro batch size | 1 |
| Effective batch size | 16 (8 DP x 1 micro x 2 GA) |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight_decay=0) |
| Max grad norm | 1.0 |
| Seed | 1234 |
| Precision | FP8 blockwise2d (e4m3fnuz, block_size=128) |
| FSDP sharding | shard_grad_op |
| Expert parallelism | EP=8 (16 local experts per rank) |
| Expert backend | Sonic (Triton GEMM) |
| MoE global expert layout | enabled |
| Gradient checkpointing | selective (attention checkpointed, MoE retained) |
| MoE activation offload | enabled (synchronous CPU offload) |
| PYTORCH_ALLOC_CONF | expandable_segments:True |

## LoRA Configuration

| Target | Rank | Alpha |
|--------|------|-------|
| Attention (q/k/v/o_proj) | 32 | 64 |
| Expert (gate_up/down) | 16 | 32 |
| Total trainable params | 576 modules |
| Adapter dtype | bfloat16 |

## Dataset

| Split | Samples | Path |
|-------|---------|------|
| Train | 1,000 | `data/build/qwen3_30b_a3b_phase1/train.tokenized.jsonl` |
| Validation | 200 | `data/build/qwen3_30b_a3b_phase1/dev.tokenized.jsonl` |

## Training Curve

### Train Loss

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

### Validation Loss

| Step | Validation Loss |
|------|----------------|
| 25   | 0.2363 |
| 50   | 0.1963 |
| 75   | 0.1870 |
| 100  | 0.1834 |
| 125  | 0.1775 |

Validation loss decreased monotonically across all checkpoints, with no sign of overfitting.

## Key Technical Decisions

### FSDP2 sharding: `shard_grad_op` instead of `full_shard`

`full_shard` causes NCCL deadlock during backward when combined with EP all-to-all. The backward all-gather (to re-materialize parameters) and the MoE all-to-all backward operate on overlapping process groups (DP=8 and EP=8 share the same 8 ranks), causing collective ordering conflicts. `shard_grad_op` avoids the backward all-gather entirely.

### Synchronous MoE activation offload

Asynchronous offload using `torch.cuda.Stream` + `record_event` causes `HSA_STATUS_ERROR_EXCEPTION` (0x1016) on ROCm during backward. The synchronous variant (`tensor.to("cpu", non_blocking=False)`) avoids this ROCm-specific issue while still reducing peak GPU memory enough to prevent OOM at 12K sequence length.

### Sonic ModuleList support

Qwen3-Coder uses `nn.ModuleList` for experts (not packed `gate_up_proj`/`down_proj` tensors). `_SonicLocalExperts.__init__` was extended to extract and repack weights from individual expert modules into the Sonic-expected `w1`/`w2` layout.

## GPU Memory Profile

| Phase | Memory per GPU |
|-------|---------------|
| After model load + FSDP2 wrap | ~136 GB |
| During forward (peak) | ~173 GB |
| During backward (peak) | ~185 GB |
| Headroom | ~7 GB / 192 GB |

## Merged Model

The best checkpoint (step 100, val_loss=0.1834) was merged into the base model using the GEAK LoRA merge pipeline. All 18,624 weight deltas (192 attention + 18,432 expert) were applied to produce a standard HF-format checkpoint.
