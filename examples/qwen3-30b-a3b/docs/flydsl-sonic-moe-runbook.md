# Qwen3-30B-A3B FlyDSL SonicMoE runbook

Reproduce Megatron Qwen3-30B-A3B with **FlyDSL** as the SonicMoE expert
backend. Router, permute, and EP all-to-all stay in Megatron.

Operator notes (ms/call): [`flydsl-sonic-moe-expert-brief.md`](flydsl-sonic-moe-expert-brief.md).
AITER Triton path: [`sonic-moe-aiter-expert-kernel-path.md`](sonic-moe-aiter-expert-kernel-path.md).

---

## 0. What this path is

| Item | Value |
|---|---|
| Hardware | 1× 8-GPU MI350X (gfx950) |
| Parallelism | TP=1, PP=1, CP=1, **EP=8** |
| Precision | **BF16 only**. Do not set `FP8_MODE=blockwise2d` |
| Expert replace | `MOE_IMPL=sonic` → `--lumen-sonic-moe` |
| GEMM backend | `SONIC_MOE_GEMM_BACKEND=flydsl` |
| Weight layout | `SONIC_MOE_FLYDSL_NATIVE=1` (default): `[E, N, K]` |
| Local shape | `E=16`, `H=2048`, `I=768`, SwiGLU `w1` N=`2I=1536`, topk=8 |
| Mean routes/rank | `T = MBS × seq × topk / EP` (MBS=2, seq=4096 → **8192**) |

`SONIC_MOE_GROUPED_GEMM_BACKEND` is ignored (AITER Triton vs hipBLASLt only).
Default training (`SONIC_MOE_GEMM_BACKEND=triton`) is AITER `moe_pre_routed_inputs`.

---

## 1. Prerequisites

1. Lumen with `lumen/ops/moe/flydsl_grouped.py`.
2. FlyDSL tree containing `kernels/moe/sonic.py` and `sonic_backward.py`.
3. Image `zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260829-multistream`.
4. Optional production assets under `HOST_ASSET_ROOT` → `/nobackup`.

```bash
test -f "${FLYDSL_ROOT:-/home/leiwu/FlyDSL}/kernels/moe/sonic.py"
```

`run_docker.sh` mounts `$FLYDSL_ROOT` at `/workspace/FlyDSL`, caches JIT in
`lumen-qwen3-flydsl-cache` (`/root/.flydsl`), and puts FlyDSL on `PYTHONPATH`.

---

## 2. Smoke (mock data)

```bash
cd /home/leiwu/Lumen/examples/qwen3-30b-a3b

MOE_IMPL=sonic \
SONIC_MOE_GEMM_BACKEND=flydsl \
FLYDSL_ROOT=/home/leiwu/FlyDSL \
FP8_MODE=bf16 \
CUDA_GRAPH_SCOPE=none \
TRAIN_STEPS=2 SEQ_LEN=1024 MBS=1 GBS=8 \
LR_WARMUP_ITERS=1 \
RUN_SUFFIX=flydsl-smoke \
./run_docker.sh
```

Expect rank 0: `> Replaced 48 Megatron MoE expert modules with SonicMoE`.
First step JIT-compiles; later steps reuse the named volume.

---

## 3. Production e2e

Attn CUDA Graph is OK (`CUDA_GRAPH_SCOPE=attn`). Do **not** graph experts
(`forward_routes_training` raises if the stream is capturing).

```bash
cd /home/leiwu/Lumen

HOST_ASSET_ROOT=/dev/shm/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
MEGATRON_LOAD_PATH=/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8 \
MOE_IMPL=sonic \
SONIC_MOE_GEMM_BACKEND=flydsl \
FLYDSL_ROOT=/home/leiwu/FlyDSL \
FP8_MODE=bf16 \
CUDA_GRAPH_SCOPE=attn \
TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
RUN_SUFFIX=flydsl-mbs2-gbs256 \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B
bash run_qwen3_30b_a3b_megatron.sh' \
bash examples/qwen3-30b-a3b/run_docker.sh
```

AITER baseline: omit `SONIC_MOE_GEMM_BACKEND=flydsl`. Host env is **not**
forwarded unless listed in `run_docker.sh --env`.

---

## 4. Launch call path

```
run_docker.sh  SONIC_MOE_GEMM_BACKEND=flydsl
  → mount FlyDSL + flydsl_grouped.py
run_qwen3_30b_a3b_megatron.sh:21-29
  MOE_IMPL=sonic → --lumen-sonic-moe
pretrain_qwen3_30b_a3b_megatron.py
  make_lumen_model_provider → LumenConfig.enable()     lumen/models/megatron.py:1171
    sonic_moe from --lumen-sonic-moe                   lumen/config.py:74, 304-305
    replace_megatron_moe_experts                       sonic_moe.py:577-585
      SonicMoEExperts: gemm_backend=flydsl, native [E,N,K]
```

---

## 5. One step: expert path

```
MoELayer.forward                                   moe_layer.py:524
  dispatch / routed_experts_compute                :469-488
    SonicMoEExperts.forward                        sonic_moe.py:357
      flydsl_pre_routed                            flydsl_grouped.py:914
        _FlyDSLNativeRoutes                        :787
          SonicMoE.forward_routes_training         FlyDSL sonic.py:3200
            gemm1 + SwiGLU + gemm2 + score
          retain sorter tensors
  combine                                          moe_layer.py:496

backward:
  _FlyDSLNativeRoutes.backward                     flydsl_grouped.py:878
    sonic_moe_backward_routes                      FlyDSL sonic_backward.py:6567
```

Shapes at the FlyDSL boundary: `hidden [T,2048] bf16`, `counts [16] int32`,
`scores [T] fp32`, native `w1 [16,1536,2048]`, `w2 [16,2048,768]`.
`token_indices=arange(T)`, `expert_offsets=cu_seqlens`,
`token_indices_identity=True`.

If native-routes conditions fail, `_FlyDSLPreRouted` (`flydsl_grouped.py:498`)
uses gemm1/gemm2 + grouped NN/TN. Qwen3 E16 BF16 should not take that fallback.

---

## 6. Kernel check (1 GPU)

```bash
pytest tests/modules/test_sonic_moe.py::test_flydsl_sonic_pre_routed_matches_torch -s
pytest tests/ops/test_flydsl_grouped.py -s
pytest tests/kernels/bench_sonic_moe_aiter_vs_flydsl.py -s
```

Numerical gates are FlyDSL vs PyTorch. The bench times AITER vs FlyDSL
`SonicMoEExperts` on already-routed E=16 rows; it does not `assert_close` them.

---

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `FlyDSL SonicMoE not found` | `FLYDSL_ROOT` missing `kernels/moe/sonic.py` |
| `SONIC_MOE_GEMM_BACKEND must be 'triton' or 'flydsl'` | Typo; hipBLASLt belongs in `SONIC_MOE_GROUPED_GEMM_BACKEND` |
| `import kernels.moe.sonic` / `flydsl` fails | `PYTHONPATH` missing FlyDSL root or `python/` + MLIR |
| `forward_routes_training does not support graph capture` | Expert region captured; use `CUDA_GRAPH_SCOPE=attn` or `none` |
| First-step OOM ~100 GiB | Adapter already sets `max_cached_workspaces=1` |
| `FP8_MODE=blockwise2d` + flydsl | Unsupported; FlyDSL path is BF16 |
| First step slow | JIT into `lumen-qwen3-flydsl-cache` |

Not this path: FSDP (`pretrain.py` still rejects non-triton backend), MORI EP.
