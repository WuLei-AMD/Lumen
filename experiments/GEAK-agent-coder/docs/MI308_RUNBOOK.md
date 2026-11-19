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

---

## From-Scratch Kernel Benchmark

### GPU topology

```text
GPU 0:   vLLM model serving (inference only, never used for kernel eval)
GPU 2-7: Kernel evaluation (6 parallel workers, each pinned to one GPU)
GPU 1:   Reserved (not used by benchmark)
```

### Step 1: Launch vLLM server

The server must use `--tool-call-parser qwen3_coder` — other parsers (hermes,
etc.) silently break tool calling for Qwen3-Coder models.

```bash
docker exec -d geak-sft-32k bash -c '
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
python3 -m vllm.entrypoints.openai.api_server \
    --model Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-2epoch \
    --served-model-name sft2f \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --enforce-eager \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
'
```

Wait for health check:

```bash
for i in $(seq 1 180); do
    curl -s http://localhost:8000/health > /dev/null 2>&1 && break
    sleep 5
done
```

Verify model responds correctly before benchmark:

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"sft2f","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10}'
```

If model outputs garbage ("!!!!"), the checkpoint is broken — do not proceed.

### Step 2: Run benchmark

```bash
docker exec geak-sft-32k bash -c '
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent
export PYTHONPATH="src:.:scripts:/home/danyzhan/Lumen-RL/experiments/sandbox:${PYTHONPATH}"
export EVAL_GPU_IDS="2,3,4,5,6,7"

# Clean previous sessions
rm -rf /home/danyzhan/held-out-benchmark-aiter/receipts/geak-flow-sft2f/

python3 -u scripts/run_aiter_benchmark_geak.py \
    --model-label sft2f \
    --model-name sft2f \
    --max-turns 200
'
```

### Step 3: Multi-model pipeline (SFT-2F → Base → SFT-4F)

Full pipeline runs sequentially, switching vLLM between models:

```bash
docker exec -d geak-sft-32k bash -c '
export PYTHONUNBUFFERED=1
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent
export PYTHONPATH="src:.:scripts:/home/danyzhan/Lumen-RL/experiments/sandbox:${PYTHONPATH}"
export EVAL_GPU_IDS="2,3,4,5,6,7"
LOG=/home/danyzhan/geak_benchmark_sftf.log

# ===== SFT-2F =====
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
python3 -m vllm.entrypoints.openai.api_server \
    --model Zhangdanyang/Qwen3-Coder-30B-A3B-SFT-TH-2epoch \
    --served-model-name sft2f \
    --tensor-parallel-size 1 --max-model-len 262144 \
    --enforce-eager --dtype bfloat16 --trust-remote-code \
    --gpu-memory-utilization 0.95 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder &
# Wait for ready...
for i in $(seq 1 180); do curl -s http://localhost:8000/health > /dev/null 2>&1 && break; sleep 5; done

rm -rf /home/danyzhan/held-out-benchmark-aiter/receipts/geak-flow-sft2f/
python3 -u scripts/run_aiter_benchmark_geak.py \
    --model-label sft2f --model-name sft2f --max-turns 200 >> $LOG 2>&1

# ===== Switch to Base =====
pkill -f vllm || true; sleep 15
# Launch Base model, same flags but --model Qwen/Qwen3-Coder-30B-A3B-Instruct --served-model-name base
# Run benchmark with --model-label base --model-name base

# ===== Switch to SFT-4F =====
pkill -f vllm || true; sleep 15
# Launch SFT-4F model, same flags but --model Zhangdanyang/...SFT-TH-4epoch --served-model-name sft4f
# Run benchmark with --model-label sft4f --model-name sft4f
'
```

### Benchmark task orchestration

The benchmark script `scripts/run_aiter_benchmark_geak.py` orchestrates:

```text
1. Load baselines
   - Read aiter_baseline_results.json (100 tasks, frozen AITER perf)
   - Filter to passing tasks only

2. Build cases
   - Generate benchmark_cases.yaml from held-out artifacts
   - Each task: operator type, shape, task_runner.py, config.yaml

3. Parallel execution (6 workers via ThreadPoolExecutor)
   - Each worker gets one GPU (EVAL_GPU_IDS split)
   - Tasks assigned round-robin to workers
   - Each task runs independently:

     a. Create GEAK session (GEAKStatefulTool.create)
        - clear_kernel=True (from-scratch mode)
        - establish_baseline=True
        - Materialize workspace with empty kernel.py skeleton

     b. Run ToolAgentLoop (200 turns max)
        - System prompt: expert GPU kernel engineer instructions
        - User prompt: operator type, baseline perf, target 90%
        - Tools: read_file, write_file, evaluate
        - On each successful evaluate: save snapshot (kernel_sp{X}_t{N}.py)
        - On speedup >= 0.9x: inline re-benchmark → if confirmed, TERMINATED
        - On generation/tool error: catch, feed back as error context, continue
        - On max turns: TERMINATED

     c. Post-task verification
        - Check if kernel_best.py exists (snapshot or ever_correct)
        - Restore kernel_best.py → kernel.py
        - Run correctness check
        - Re-benchmark: 3-run median for both candidate and baseline
        - Report final verified speedup

4. Results
   - Log: sp={X}x turns={N} tools={N} FAST|OK|FAIL
   - Sessions saved: receipts/geak-flow-{label}/sessions/
   - Snapshots: sessions/*/workspace/snapshots/kernel_sp*.py
```

### Prompt format

The benchmark uses natural language chat format (matching SFT-f training data):

```text
System: "You are an expert GPU kernel engineer writing Triton kernels
         for AMD MI300X (gfx942). You have access to a GEAK sandbox tool
         with actions: list_files, read_file, write_file, evaluate. ..."

User:   "Write a high-performance Triton kernel for the 'gemm' operator
         on AMD MI300X (gfx942).
         Task: {task_id}
         Baseline performance: {bl_ms} ms
         Target: achieve at least 90% of baseline ({target_ms} ms or faster).
         Steps: 1) Read task_runner.py ..."
```

NOT the old JSON `geak_kernel_sft_v1` format. The `sft_compat` flag is NOT used.

### Monitoring

```bash
# Quick status
grep -c "sp=" /home/danyzhan/geak_benchmark_sftf.log
grep "sp=" /home/danyzhan/geak_benchmark_sftf.log | grep -v "sp=0.00x"

# Snapshot count
find receipts/geak-flow-sft2f/sessions/ -name "kernel_best.py" -path "*/snapshots/*" | wc -l

# vLLM health
curl -s http://localhost:8000/health
curl -s http://localhost:8000/metrics | grep "num_requests_running"

# Worker processes
ps aux | grep "run_aiter_benchmark" | grep -v grep | wc -l
```

### Manual re-benchmark of snapshots

For tasks that exited early (exception at turns <200) but have kernel_best.py:

```bash
docker exec geak-sft-32k python3 -u -c "
import json, subprocess, shutil, os
from pathlib import Path

FLOW = Path('/home/danyzhan/held-out-benchmark-aiter/receipts/geak-flow-sft2f/sessions')
BASELINES = json.load(open('/home/danyzhan/held-out-benchmark-aiter/receipts/aiter_baseline_results.json'))

gpu_id = 2
env = {**os.environ, 'HIP_VISIBLE_DEVICES': str(gpu_id)}

for sess in sorted(FLOW.iterdir()):
    meta_f = sess / 'workspace' / 'metadata.json'
    if not meta_f.exists(): continue
    meta = json.load(open(meta_f))
    tid = meta.get('task_id', '')

    kb = sess / 'workspace' / 'snapshots' / 'kernel_best.py'
    if not kb.exists(): continue

    ws = sess / 'workspace'
    shutil.copy2(kb, ws / 'kernel.py')

    # Correctness
    cr = subprocess.run(['python3', 'scripts/task_runner.py', 'correctness'],
        cwd=str(ws), capture_output=True, text=True, timeout=120, env=env)
    if cr.returncode != 0 or 'OK' not in cr.stdout: continue

    # Performance (3 runs)
    cand_times = []
    for _ in range(3):
        r = subprocess.run(['python3', 'scripts/task_runner.py', 'performance'],
            cwd=str(ws), capture_output=True, text=True, timeout=120, env=env)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if 'Perf:' in line:
                    cand_times.append(float(line.split('Perf:')[1].split('ms')[0].strip()))

    bl_ms = next((r['perf_ms'] for r in BASELINES if r['task'] == tid), 0.0)
    if cand_times and bl_ms > 0:
        cand_median = sorted(cand_times)[len(cand_times)//2]
        speedup = bl_ms / cand_median
        print(f'{tid}: {speedup:.2f}x')
"
```

### Known issues and fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong tool-call parser | 0 tools per turn, text output | `--tool-call-parser qwen3_coder` |
| Tool execution exception | Task exits at 6/50 turns | try/except in runtime.py tool execution |
| Generation exception | Task exits early | try/except in runtime.py backend.generate |
| Post-task misses snapshots | sp=0.00x but snapshot has 1.89x | Check kernel_best.py exists, not just ever_correct |
| GPU zombie after vLLM crash | Next vLLM OOM | `docker restart geak-sft-32k` |
| Autotune measurement variance | 6x noise | Frozen configs in autotune_freeze.json |
| Model outputs "!!!!" | Broken SFT checkpoint | Retrain model, check adapter merge |
