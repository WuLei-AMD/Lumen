# GEAK Kernel SFT Dataset Construction Runbook

## 0. Agent Mission

本 runbook 交给数据构造 Agent 执行。目标是从零搭建可复现环境，使用：

- Lumen-RL workflow branch：`dev/multi-loop-agent`
- GEAK branch：`main`
- 真实 AMD GPU 上的 compile、correctness、performance 和 independent verify

构造可用于 Qwen3-Coder-30B-A3B LoRA/SFT 的 Kernel Coding 数据。

本 runbook 的主控制面是 Lumen-RL MultiTune。GEAK `main` 提供上游 workspace
materialization、GPU lock、验证语义、知识和任务基础设施，并保持源码 checkout
可追溯。GEAK 原生 Claude Code Workflow 的 `exp/` artifact 可以作为补充来源，但
只有在映射到相同 schema、补齐模型可见上下文并重新 independent verify 后才能
合并；不能与 MultiTune trajectory 直接拼接计数。

训练目标是单轮 Kernel Coding Policy：

```text
真实可见的 contract/source/feedback
→ 一个可应用的 unified diff
```

Agent 不得把普通日志行、Engineer 自报结果、未验证 patch 或 Harness 修改当作正向
SFT 样本。

详细 schema、数据比例和模型验收原则见同目录：

```text
GEAK-SFT-Model.md
```

### 0.1 Git 代码库地址

```text
Lumen-RL repository:
https://github.com/ZhangDanyang-AMD/Lumen-RL.git

Required branch:
dev/multi-loop-agent

Branch URL:
https://github.com/ZhangDanyang-AMD/Lumen-RL/tree/dev/multi-loop-agent

GEAK repository:
https://github.com/AMD-AGI/GEAK.git

Required branch:
main

Branch URL:
https://github.com/AMD-AGI/GEAK/tree/main
```

Agent 必须从以上 repository URL 获取代码，并在环境 manifest 中保存 remote URL、
branch 和 commit SHA。禁止使用来源不明的镜像仓库或未记录的 fork。

## 1. 最终交付物

最终发布物必须是可直接上传到 Hugging Face Dataset repository 的自包含目录，布局
对齐 [`Zhangdanyang/final-hip-gfx950-all-v2`](https://huggingface.co/datasets/Zhangdanyang/final-hip-gfx950-all-v2)：

```text
hf_dataset_root/
├── .gitattributes
├── README.md                    # Hugging Face dataset card，含 YAML metadata
├── dataset_card.md
├── manifest.json
├── input_manifest.yaml
├── samples.jsonl               # verified Train + Dev；每行带显式 split
├── rejected.jsonl
├── artifacts/
│   └── <sample_id>/
│       ├── parent/
│       │   ├── kernel.<ext>
│       │   ├── config.yaml
│       │   ├── metadata.json
│       │   └── scripts/task_runner.py
│       └── candidate/
│           ├── kernel.<ext>
│           ├── config.yaml
│           ├── metadata.json
│           └── scripts/task_runner.py
├── metadata/
│   ├── catalog.yaml
│   ├── environment.json
│   ├── split_manifest.json
│   └── support_matrix.json
├── reports/
│   ├── collection_report.json
│   ├── quality_report.json
│   ├── leakage_report.json
│   ├── coverage_report.json
│   └── trust_report.json
└── checksums.sha256
```

`samples.jsonl` 是 Hugging Face 发布时的唯一正样本索引，不再发布
`processed/train.jsonl`、`processed/dev.jsonl` 或内容寻址 `blobs/` 作为用户接口。
内部 ETL 可以继续使用这些 staging 路径，但发布前必须转换为上述布局。每条
`samples.jsonl` 记录至少包含：

```json
{
  "sample_id": "<stable-id>",
  "split": "train | dev",
  "sample_domain": "kernel | general_coding",
  "task_type": "cold_start | profile_guided | direction_conditioned | error_recovery | regression_balance | general_coding_replay",
  "parent_sources": {
    "kernel.py": {"path": "artifacts/<sample_id>/parent/kernel.py", "sha256": "..."}
  },
  "candidate_sources": {
    "kernel.py": {"path": "artifacts/<sample_id>/candidate/kernel.py", "sha256": "..."}
  },
  "artifact_paths": {
    "parent_workspace": "artifacts/<sample_id>/parent",
    "candidate_workspace": "artifacts/<sample_id>/candidate"
  },
  "output": {"patch": "..."},
  "labels": {},
  "provenance": {}
}
```

HIP 源文件使用真实扩展名（例如 `kernel.hip` 或 `kernel.cpp`），Triton 使用
`kernel.py`；禁止为了统一 schema 改写源码语言。所有 `path` 必须相对
`hf_dataset_root/`，不得出现 `/home/...`、容器绝对路径或只能在生产机解析的
symlink。`parent/` 与 `candidate/` 都必须携带可本地 replay 的 `config.yaml`、
`metadata.json` 和 `scripts/task_runner.py`。

Train 与 Dev 可以共用一个 `samples.jsonl`，但每行必须有显式 `split`，且
`README.md` 必须给出按 split 加载和过滤的示例。`general_coding_replay` 如发布，
也写入 `samples.jsonl` 并使用独立 `task_type`；不得复制为第二份数据文件。

下载 repository 后必须能仅依赖相对路径读取和 replay：

```python
import json
from pathlib import Path

root = Path(".")
sample = json.loads((root / "samples.jsonl").open().readline())
parent_code = (root / sample["parent_sources"]["kernel.py"]["path"]).read_text()
candidate_code = (root / sample["candidate_sources"]["kernel.py"]["path"]).read_text()
candidate_workspace = root / sample["artifact_paths"]["candidate_workspace"]
```

实际代码 key 由源文件相对路径决定，loader 不得硬编码只有 `kernel.py`。
`.gitattributes` 必须让大 artifact 使用 Hugging Face/Xet 或 Git LFS，且
`checksums.sha256` 覆盖 pointer 解引用后的真实文件内容。

每条正样本必须能从 `manifest.json` 和相对 artifact path 追溯到：

1. Lumen-RL 与 GEAK Git SHA；
2. 原始 run、round、candidate 和 Engineer；
3. 模型真实看到的 parent source 和 prompt；
4. patch 原始字节；
5. 独立 verify 的 compile/correctness/performance receipt；
6. 准确 GPU SKU、gfx、ROCm、compiler 和容器 digest；
7. split、source lineage 和 implementation family。

Held-out 不得放入上述 Train/Dev Hugging Face repository。它必须发布到访问权限
独立的私有评测 repository：

```text
hf_held_out_root/
├── README.md
├── manifest.json
├── tasks/
│   ├── kernel.jsonl             # Kernel contract/初始源码引用，无 target patch
│   └── general_coding.jsonl     # 独立 Coding benchmark tasks
├── artifacts/
│   ├── kernel/<task_id>/initial/
│   └── general_coding/<task_id>/initial/
├── protected/                   # 仅评测服务可见，模型与训练流水线不可见
│   ├── kernel_harness_oracle/
│   └── general_coding_solution_test_oracle/
├── receipts/
│   ├── kernel/{base,sft}/
│   └── general_coding/{base,sft}/
├── reports/
│   ├── kernel_evaluation_report.json
│   ├── general_coding_evaluation_report.json
│   ├── retention_gate_report.json
│   └── leakage_report.json
└── checksums.sha256
```

模型可见的 Held-out namespace 禁止出现 `candidate/`、`candidate_sources`、参考
solution/优化 patch、test patch、oracle 或训练 trajectory。General Coding 的
solution patch、test patch 和 oracle 只能存在于使用独立 credentials 的
`protected/` namespace 或评测服务端；模型可见 manifest 仅保存 SHA256。若 Hugging
Face 权限不能保证模型和训练任务无法读取 `protected/`，则 protected artifacts
不得上传 HF，只保存在评测服务端。Kernel 与 General Coding 必须使用不同 task
manifest、artifact namespace、receipt 和 report；二者不能合并成一个成功率。

## 2. 不可违反的规则

### 2.1 数据真实性

- 只将 independent verify 通过的 patch 作为正向 SFT target。
- `worker_result.json`、Engineer report 和自报 speedup 只能进入
  `auxiliary_claims`。
- `parent_source` 必须是模型生成该 patch 时真实看到的源码。
- Profile 必须来自同一个 parent source 和同一环境。
- Direction 必须在 Engineer 执行前保存。
- Cold-start 样本必须由真正无 profile、无 direction 的运行产生，禁止事后删除
  字段伪装。
- Error-recovery 的输入必须是失败后的源码和当时真实错误。
- Patch 必须能在干净 workspace 中应用。

### 2.2 执行边界

- 只允许修改任务声明的 kernel source。
- Harness、oracle、tests、config、metadata 和隐藏 case 只读。
- 禁止 symlink 越界、修改验证命令、固定输出、缓存答案、调用 reference 代替目标
  kernel。
- Correctness 通过前禁止将 performance 结果标为有效。
- gfx942 的验证结果不得替代 gfx950，反之亦然。
- 必须记录准确 GPU SKU；不能把 MI308 latency 标记成 MI300 latency。
- Native MXFP8/MXFP6/MXFP4 只允许 gfx950。gfx942 应在生成和 GPU 执行前拒绝。

### 2.3 Split 与泄漏

- 在生成数据前冻结 train/dev/held-out 分组。
- 按 source lineage、implementation family、operator contract 和 shape cluster
  分组切分。
- 同一 kernel 的相邻 patch、多轮轨迹、语言翻译版和近重复 shape 不得跨 split。
- Held-out 内容不得进入 prompt、Skill、KB、retrieval corpus 或训练数据。
- 隐藏 seed/case 只能保存在受保护 metadata 中，不能进入模型可见上下文。

## 3. 阶段目标与数量

### 3.1 阶段一：Triton、HIP × gfx942、gfx950

四个组合：

```text
Triton × gfx942
Triton × gfx950
HIP    × gfx942
HIP    × gfx950
```

目标：

```text
Verified Kernel SFT： 3,000–4,000 条
每个组合：              750–1,000 条
通用 Coding replay：    Phase 1 固定 500 条
Held-out tasks：        每个组合 40–80 个，共 160–320 个
```

### 3.2 阶段二：加入 FlyDSL、Gluon

新增：

```text
FlyDSL × gfx942
FlyDSL × gfx950
Gluon   × gfx942
Gluon   × gfx950
```

八组合最终目标：

```text
Verified Kernel SFT 总量： 6,000–8,000 条
每个有效组合：               750–1,000 条
阶段二新增：                 约 3,000–4,000 条
通用 Coding replay 总量：     700–1,200 条
Held-out tasks：             每个组合 40–80 个，共 320–640 个
```

如果 FlyDSL/Gluon 在某个架构上没有真实稳定执行链路，将组合标记为
`unsupported` 或 `not_ready`。禁止生成虚假样本配平。

### 3.3 每个组合的任务类型

目标比例：

```text
Cold-start             15%
Profile-guided         15%
Direction-conditioned  45%
Error-recovery         15%
Regression-balance     10%
```

不得复制样本凑比例。先用 pilot 计算每次 run 的有效样本产率，再决定实际 run 数。
报告中必须同时提供：

- raw run 数；
- candidate 数；
- independent verified candidate 数；
- 去重前/后样本数；
- 独立 task 和 source lineage 数；
- 有效 Assistant token 数。

## 4. 从零搭建机器环境

### 4.1 硬件与系统前提

最低要求：

- Linux；
- gfx942 或 gfx950 AMD Instinct GPU；
- ROCm 6+；
- Docker 可访问 `/dev/kfd` 和 `/dev/dri`；
- Python 3.10+；
- Git、tmux；
- `rocprof-compute`、`rocprofv3` 或 `rocprof` 中至少一个 profiler。

推荐每台机器至少两张 GPU：

```text
GPU 0：vLLM generator model
GPU 1：GEAK compile/correctness/performance
```

若只有一张 GPU，不得同时运行模型服务与 kernel benchmark。应使用远程模型 endpoint，
或在采数前停止模型服务并确认 GPU idle。

执行并保存输出：

```bash
uname -a
python3 --version
git --version
docker --version
rocminfo
rocm-smi
hipcc --version
```

硬门禁：

- `rocminfo` 必须显示预期 `gfx942` 或 `gfx950`；
- `rocm-smi` 必须显示准确产品名和物理 GPU 数；
- Docker 容器内必须能读取同样的 GPU；
- 生产 benchmark GPU 不得被其他进程占用。

Docker GPU smoke test：

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  rocm/dev-ubuntu-24.04:latest rocminfo
```

镜像 tag 只用于第一次验证。生产环境必须将实际镜像保存为不可变 digest。

### 4.2 创建工作目录

以下路径可调整，但必须使用绝对路径：

```bash
export WORK_ROOT="$HOME/geak-sft-work"
export LUMEN_ROOT="$WORK_ROOT/Lumen-RL"
export GEAK_HOME="$WORK_ROOT/GEAK"
export STAGING_DATASET_ROOT="$WORK_ROOT/datasets/geak-kernel-sft-v1-staging"
export HF_DATASET_ROOT="$WORK_ROOT/hf/geak-kernel-sft-v1"
export HF_HELD_OUT_ROOT="$WORK_ROOT/hf/geak-kernel-sft-v1-held-out"
export RUN_ROOT="$WORK_ROOT/multi_tune_runs"
mkdir -p "$WORK_ROOT" "$STAGING_DATASET_ROOT" "$HF_DATASET_ROOT" \
  "$HF_HELD_OUT_ROOT" "$RUN_ROOT"
```

### 4.3 克隆并锁定正确分支

```bash
git clone https://github.com/ZhangDanyang-AMD/Lumen-RL.git "$LUMEN_ROOT"
git -C "$LUMEN_ROOT" switch dev/multi-loop-agent
git -C "$LUMEN_ROOT" pull --ff-only

git clone https://github.com/AMD-AGI/GEAK.git "$GEAK_HOME"
git -C "$GEAK_HOME" switch main
git -C "$GEAK_HOME" pull --ff-only
```

记录状态：

```bash
git -C "$LUMEN_ROOT" status --short --branch
git -C "$LUMEN_ROOT" rev-parse HEAD
git -C "$GEAK_HOME" status --short --branch
git -C "$GEAK_HOME" rev-parse HEAD
```

硬门禁：

- Lumen-RL 当前 branch 必须是 `dev/multi-loop-agent`；
- GEAK 当前 branch 必须是 `main`；
- 首次环境冻结时两个仓库都必须无未记录修改；
- SHA 必须写入 environment manifest；
- 采数期间禁止自动 `git pull`，版本升级必须创建新 dataset version。

### 4.4 创建 Python 环境

```bash
export MULTITUNE_ROOT="$LUMEN_ROOT/experiments/multi-tune-agent"
cd "$MULTITUNE_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install PyYAML requests "pytest>=8.0"
# Install the checked-out GEAK main branch itself. Do not let pip silently
# resolve a different GEAK commit from the network.
GEAK_SKIP_BOOTSTRAP=1 python3 -m pip install -e "$GEAK_HOME"
python3 -m pip install -e . --no-deps
```

上面的 `--no-deps` 是有意的：MultiTune 的 `pyproject.toml` 将 GEAK 声明为未锁定
Git URL；直接执行 `pip install -e ".[test]"` 可能安装与本地 `GEAK/main` SHA 不同的
版本。只有临时开发环境可以使用简化安装：

```bash
GEAK_HOME="$GEAK_HOME" python3 -m pip install -e ".[test]"
```

验证：

```bash
python3 -c "import geak, geak_utils, multi_tune_agent, requests, yaml"
python3 -c "import geak, pathlib; print(pathlib.Path(geak.__file__).resolve())"
pytest -q
cd "$GEAK_HOME"
python3 -m pytest -q geak/test_bootstrap.py kernel_workflow/scripts/tests
cd "$MULTITUNE_ROOT"
```

打印的 `geak.__file__` 必须位于当前 `$GEAK_HOME`。如果指向
`site-packages` 中的另一份副本，停止并修正环境。

不要忽略失败测试。任何失败都必须记录并修复后才能开始生产采数。

### 4.5 创建机器专用配置

不要直接依赖仓库中 `/home/danyzhan/...` 的默认路径。复制配置并改成当前机器的
绝对路径：

```bash
cd "$MULTITUNE_ROOT"
cp configs/mi300x.yaml "configs/dataset-$(hostname).yaml"
export LUMEN_CODE_CONFIG="$MULTITUNE_ROOT/configs/dataset-$(hostname).yaml"
```

至少检查：

```yaml
geak_root: /absolute/path/to/GEAK
cases_path: ../cases/examples_cases.yaml
trajectory_root: /absolute/path/to/multi_tune_runs
base_url: http://127.0.0.1:8000/v1
model: Qwen/Qwen3-Coder-Next
gpu_ids: "1"
bootstrap_enabled: true
bootstrap_auto_promote: true
aiter_root: /absolute/path/to/aiter
generated_template_root: ../examples/tasks/generated
baseline_repeats: 3
max_rounds: 3
engineers_per_round: 3
candidate_floor: 1.0
min_improvement: 0.02
keep_sessions: true
```

生产采数建议：

- `keep_sessions: true`；
- baseline 至少重复 3 次；
- kernel GPU 与 model GPU 不重叠；
- timeout 必须容纳 HIP 编译；
- config 原始字节与 SHA256 必须写入每个 run provenance。

设置环境：

```bash
export GEAK_HOME="$GEAK_HOME"
export LUMEN_CODE_GEAK_ROOT="$GEAK_HOME"
export LUMEN_CODE_CONFIG="$LUMEN_CODE_CONFIG"
export GEAK_CONTAINER_NAME=geak-phase1-vllm
export AITER_HOME="${AITER_HOME:-$HOME/aiter}"
export HIP_VISIBLE_DEVICES=1
```

不要将 API key、token 或私有模型凭据写入 config、trajectory 或 dataset。

### 4.6 启动 generator model

先创建缓存：

```bash
mkdir -p "$HOME/.cache/huggingface"
```

首次创建容器：

```bash
docker run -d --name geak-phase1-vllm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc=host --network=host --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$WORK_ROOT:$WORK_ROOT" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai-rocm:v0.15.0 \
  Qwen/Qwen3-Coder-Next-FP8 \
  --served-model-name Qwen/Qwen3-Coder-Next \
  --max-model-len 32000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 8000
```

如果容器已存在：

```bash
docker start geak-phase1-vllm
docker logs -f geak-phase1-vllm
```

必须记录：

```bash
docker inspect geak-phase1-vllm
docker image inspect vllm/vllm-openai-rocm:v0.15.0
```

注意：示例镜像和模型是当前 MultiTune 已验证配置，不是 SFT target checkpoint。
更换 generator model、chat template、decoding 参数或镜像时必须创建新的 provenance
版本，不能静默混入同一数据版本。

### 4.7 服务与 GPU 预检

```bash
cd "$MULTITUNE_ROOT"
source .venv/bin/activate
multi-tune --config "$LUMEN_CODE_CONFIG" probe
```

`configured_model_present` 必须为 `true`。

运行一个冻结的 smoke case：

```bash
multi-tune --config "$LUMEN_CODE_CONFIG" run \
  --case dense-gemm-fp16 \
  --request "Smoke test only; preserve correctness." \
  --stream
```

Smoke run 必须证明：

- 模型 endpoint 可用；
- GEAK workspace 可 materialize；
- GPU lock 生效；
- compile、correctness、performance 可运行；
- original task 未被直接修改；
- run 下产生 `trajectory.jsonl` 和 `summary.json`。

Smoke run 不得直接计入生产数据。

## 5. 生产采数前必须补齐的能力

当前 `dev/multi-loop-agent` 已有普通 `trajectory.jsonl`/`summary.json` 和优化流程，
但不能仅凭这些文件可靠恢复所有 SFT 字段。以下能力是生产采数前硬门禁。

### 5.1 Append-only SFT collector

在不改变评测语义的前提下实现：

```text
<run_dir>/sft_events.jsonl
<run_dir>/sft_manifest.json
<run_dir>/round_<N>/plan.json
<run_dir>/round_<N>/candidates.jsonl
<run_dir>/environment.json
```

`sft_events.jsonl` 每条至少包含：

```json
{
  "schema_version": "geak_sft_event_v1",
  "run_id": "...",
  "timestamp": 0,
  "role": "engineer",
  "phase": "agent_loop",
  "round": 1,
  "candidate_id": "...",
  "prompt": {},
  "response": {},
  "visible_context_hashes": {}
}
```

要求：

- append-only；
- 多 Engineer 并发写安全；
- Collector 写失败必须使 run 标记为 dataset-ineligible；
- 终端/UI subscriber 失败可以降级，但 provenance 失败不能静默吞掉；
- secret 和隐藏评测内容进入受保护 metadata，不进入模型可见 prompt。

### 5.2 Content-addressed source 与 patch

Engineer 执行前保存 parent source 原始字节：

```text
<staging_dataset_root>/blobs/sha256/<first-two>/<full-sha256>
```

同时保存：

- `source_hash`；
- `baseline_source_hash`；
- `patch_hash`；
- path、mode、size；
- workspace HEAD；
- allowed modification paths。

只保存格式化后的源码不够，必须保存模型实际看到的原始字节。

### 5.3 独立 verify receipt

每个 candidate 都要结构化保存：

- patch apply 状态；
- compile command、cwd、env allowlist、exit code、stdout/stderr；
- correctness cases、seed、tolerance、结果；
- 每次原始 baseline/candidate latency；
- warmup、repeat、timer、聚合方法；
- variance、outlier、contention；
- GPU identity 和 kernel launch/engagement 证明；
- verify workspace/session；
- rejection reason。

Engineer 的 `evaluate` 结果不能替代 independent verify。

### 5.4 批量任务生成接口

仓库中的 `cases/generation_requests.yaml` 目前只是 manifest 示例；
`multi-tune generate --manifest ...` 还未实现。规模化采数前应实现等价的非交互接口。

目标接口：

```bash
multi-tune --config "$LUMEN_CODE_CONFIG" generate \
  --manifest cases/phase1-generation-requests.yaml \
  --output-catalog cases/phase1-cases.yaml \
  --stream
```

接口必须：

1. 验证完整 contract；
2. 在生成前执行 architecture/format hard gate；
3. 优先复用已通过 GPU gate 的 trusted template；
4. 缺模板时生成 draft；
5. 按 compile → correctness → performance 验证；
6. 只注册全部通过的任务；
7. 支持幂等 resume；
8. 不把 failed draft 注册成任务；
9. 每个 request 保存状态、错误和 provenance；
10. GPU 工作继续通过 GEAK lock 串行。

在该接口实现前，只能用 interactive menu `[2]` 逐条生成 pilot，禁止声称已支持批量
生产。

### 5.5 五类 task mode

Workflow 必须显式支持并记录：

```text
cold_start
profile_guided
direction_conditioned
error_recovery
regression_balance
```

每个 mode 必须在 Engineer 调用前冻结可见字段。不能依靠 ETL 猜测 task type。

最低行为：

- `cold_start`：不生成、不注入 profile/direction；
- `profile_guided`：只注入 parent source 对应 profile；
- `direction_conditioned`：先持久化 TechLead direction，再调用 Engineer；
- `error_recovery`：输入失败后的源码和精确错误；
- `regression_balance`：输入 per-case regression report 与冻结约束。

### 5.6 ETL 和审计工具

至少实现：

```text
scripts/build_sft_dataset.py
scripts/validate_sft_dataset.py
scripts/audit_sft_leakage.py
scripts/report_sft_coverage.py
```

所有工具必须：

- 读取 raw artifact，不修改 raw；
- 幂等；
- 接受固定 input manifest；
- 输出版本和 SHA256；
- 遇到缺失字段时拒绝样本，不得猜测；
- 为每个 rejected sample 写结构化 reason。

### 5.7 Collector 单元与集成测试

开始 pilot 前至少覆盖：

- 并发 append 不丢事件；
- source/patch hash 稳定；
- prompt 中不可见字段不会被 ETL 加入；
- failed/unverified candidate 不进入正样本；
- verify 标签覆盖 Engineer claim；
- patch 对 parent source 可应用；
- source lineage group split 不泄漏；
- hidden case/seed 不出现在 prompt；
- Harness 修改导致拒绝；
- 中断后 resume 不重复计数；
- gfx/format 不兼容在 GPU 执行前失败；
- manifest/checksum 可重新生成并一致。

运行：

```bash
cd "$MULTITUNE_ROOT"
source .venv/bin/activate
pytest -q
```

## 6. 构建支持矩阵和任务清单

### 6.1 探测实际机器

每台 worker 生成 inventory：

```json
{
  "worker_id": "...",
  "gpu_sku": "MI308",
  "architecture": "gfx942",
  "gpu_count": 2,
  "rocm_version": "...",
  "languages": {
    "triton": {"status": "ready", "version": "..."},
    "hip": {"status": "ready", "compiler": "..."},
    "flydsl": {"status": "not_tested", "version": null},
    "gluon": {"status": "not_tested", "version": null}
  }
}
```

只有最小 compile/correctness/performance smoke 全部通过后，组合才能标记 `ready`。

### 6.2 Contract 必填字段

每个 generation request 必须写清：

- unique task ID；
- operator 和 entry point；
- language、language version、backend version；
- architecture 和目标 GPU family；
- M/N/K 或完整维度集合；
- input/output dtype；
- layout、stride、transpose；
- quantization packing、scale granularity、block size；
- output dtype 和 accumulation mode；
- correctness tolerance；
- shape regime 与 workload weight；
- modifiable files；
- task type；
- split group；
- source lineage/implementation family；
- 永久优化目标和 regression constraints。

模型不得补造 request 中不存在的维度、format 或语言。

### 6.3 Phase 1 manifest

生成 case 前，Agent 必须先读取人类可读的
[Phase 1 Case 登记与正式生产计划](./case_list.md)。该清单定义四条
language-architecture lane、当前已登记 seed、跨语言 shape/contract 复用规则、
P0/P1 优先级、split/lineage 约束和晋升门禁。

机器可读来源位于 Lumen-RL：

```text
experiments/multi-tune-agent/cases/phase1-case-candidates.yaml
experiments/multi-tune-agent/cases/phase1-aiter-seeds.yaml
```

Agent 必须按以下顺序生成，不能直接把 `planned_candidate` 当作可运行 task：

```text
case_list.md
→ phase1-case-candidates.yaml
→ phase1-aiter-seeds.yaml
→ phase1-generation-requests.yaml
→ static/GPU trust gate
→ phase1-cases.yaml
```

跨语言 case 只允许复用真实 shape、dtype、layout、scale contract 和独立 oracle。
若实现来自直接翻译，必须共享 `implementation_family_id`，并与原语言版本进入同一
split group。学生 kernel 运行时不得调用 AITER 或其他 reference backend。

创建：

```text
cases/phase1-generation-requests.yaml
cases/phase1-cases.yaml
```

先使用小规模 pilot：

```text
每组合 10–20 个独立 tasks
每种 task type 均至少 3 个
覆盖 decode、small-batch、prefill
覆盖对齐与非对齐尺寸
```

不要直接生成 3,000 个相邻 shape。先确认：

- task harness 正确；
- 四组合均能通过；
- 数据 collector 字段完整；
- verified sample yield 可接受；
- 性能噪声受控；
- 去重后仍有足够多独立 lineage。

### 6.4 从 AITER 提取真实 Case

Phase 1 的 operator、shape、dtype、layout 和 architecture support 优先从已锁定的
AITER tests、benchmarks、tuning configs 和 supported guards 中抽取，而不是让
LLM 发明 workload。

当前 AITER 来源：

```text
Repository: https://github.com/ROCm/aiter.git
Required provenance: remote URL + branch + commit SHA + file SHA256
License: MIT
```

优先来源：

```text
op_tests/test_gemm_a16w16.py
op_tests/test_batched_gemm_bf16.py
op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py
op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py
op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py
op_tests/op_benchmarks/triton/bench_gemm_a16w16.py
op_tests/op_benchmarks/triton/bench_gemm_a8w8.py
aiter/ops/triton/**/tunning/*.py
csrc/** 中带明确 gfx942/gfx950 guard 的实际实现
```

每个抽取 case 必须保存：

- AITER remote、branch、commit SHA；
- test/benchmark/config 原始相对路径及文件 SHA256；
- 原始 test ID 或参数化 case；
- operator、M/N/K 或完整 dims、dtype、layout、scale contract；
- 文件中的 architecture guard 和 support 条件；
- PyTorch/high-precision oracle 逻辑来源；
- 实际实现 backend（Triton/HIP/CK/ASM/FlyDSL），禁止仅凭文件名猜测；
- `source_lineage_id` 和 `implementation_family_id`。

使用边界：

1. AITER tests/benchmarks 提供真实 case 与独立 oracle；
2. AITER implementation 可作为只读 baseline/evidence 或经许可证记录的 seed；
3. 不能把 AITER 最优实现直接复制成 `output.patch`；
4. 如果训练目标是独立 HIP/Triton coder，target kernel 禁止在运行时调用 AITER
   reference 来绕过实现；
5. 每个 case 仍须进入隔离任务、通过 compile/correctness/performance，并由 fresh
   workspace independent verify；
6. AITER test、实现及其翻译版本必须按 lineage 分组，防止跨 split 泄漏；
7. 只有当前硬件真实执行通过的 support 才能写入数据标签。

## 7. 冻结 Split

### 7.1 分组键

推荐 group key：

```text
operator
+ normalized contract family
+ source lineage
+ implementation family
+ shape cluster
```

先分组，再分配 split。禁止先生成样本再随机按 JSON 行切分。

### 7.2 当前 gfx942 双 lane 的 Dev 与 Held-out 设计

当前批准的 Phase 1 gfx942 里程碑覆盖 `Triton × gfx942` 和 `HIP × gfx942`
两个 lane。Train 固定为 2,000 条 verified SFT samples；Train 完成并通过最终
coverage/trust audit 后，另外物化：

```text
                         Cold  Profile  Direction  Error  Regression  Total
Dev verified samples       30       30         90     30          20    200
Held-out tasks             18       18         54     18          12    120
```

两者都使用 `15% / 15% / 45% / 15% / 10%` 的 task type 配比，并在两个
gfx942 lane 间等分：

```text
Dev：      Triton 100 + HIP 100
Held-out： Triton  60 + HIP  60
```

这里的计数单位不同：

- Dev 的 200 指 200 条带 verified target patch 的 SFT 格式样本，但只用于训练期
  评估、诊断和 checkpoint selection，不参加梯度更新；
- Held-out 的 120 指 120 个独立 eval task，不是 120 条带答案的 SFT 样本；
- 一个 Held-out task 只能计入一种主 `task_type`，不得同时在多个格子重复计数；
- Profile-guided、Direction-conditioned、Error-recovery 和
  Regression-balance 的上下文必须由对应 split 的冻结任务真实产生，禁止从
  Train trajectory 删除或改写字段后伪造。

Kernel Held-out 的 `source_suite` 与五种 `task_type` 是两个正交维度。120 个任务的
初始 source mix 目标为：

```text
GEAK native benchmark-backed tasks       40
AITER-derived independent contracts      40
新构造的边界/回归/adversarial tasks       40
Total                                   120
```

source mix 可因独立任务库存调整，但必须记录书面偏差；`18/18/54/18/12` 的
task type 总配比保持不变。每个 source suite 内尽量覆盖 HIP/Triton，最终两个 lane
仍各 60 个。

#### Dev 的用途与物化

Dev 必须沿用与 Train 相同的 patch apply、compile、correctness、performance 和
fresh-workspace independent verify 门禁。它用于：

1. 每个 checkpoint 计算 patch-token + Assistant EOT 的 token-level eval loss；
2. 执行 200 条样本并报告 Apply、Compile、Correctness、Valid Optimization 和
   speedup；
3. 按五种 task type、HIP/Triton lane、operator/format/shape regime 定位退化；
4. 以执行指标作为 checkpoint selection 主依据，eval loss 只用于监控过拟合。

当前 Dev profile 固定为：

```text
configs/phase1-dev-wave-200.yaml
dataset_version = phase1-dev-wave-200-v1
allowed_case_splits = [dev]
```

Train 最终冻结并释放 GPU 1–7 后，按以下顺序执行：

```bash
cd "$MULTITUNE_ROOT"
source .venv/bin/activate

mkdir -p /home/danyzhan/phase1_control/dev-wave-v1
printf 'tasks: []\n' \
  > /home/danyzhan/phase1_control/dev-wave-v1/empty-base.yaml

python scripts/run_parallel_generation.py \
  --config configs/mi300x.yaml \
  --manifest /home/danyzhan/phase1_control/splits/v4/generation-requests.yaml \
  --base-catalog /home/danyzhan/phase1_control/dev-wave-v1/empty-base.yaml \
  --output-catalog /home/danyzhan/phase1_control/dev-wave-v1/production-cases.yaml \
  --work-root /home/danyzhan/phase1_control/dev-wave-v1/generation \
  --split dev

python scripts/run_phase1_production_controller.py \
  --profile-file configs/phase1-dev-wave-200.yaml \
  --cases /home/danyzhan/phase1_control/dev-wave-v1/production-cases.yaml \
  --trajectory-root /home/danyzhan/phase1_control/dev-wave-v1 \
  --dataset-root /home/danyzhan/geak_sft_dataset/phase1-dev-wave-200-v1 \
  --config configs/mi300x.yaml \
  --gpus 1-7
```

`phase1-dev-wave-200.yaml` 是 `checkpoint_sample_count=0` 的独立 profile；
controller 不得隐式载入 Train seed manifest。Train catalog 全部为 `split_group=train`，
也不得作为 Dev 的 base catalog。

Dev 发布门禁：

```text
sample count                       = 200
task type counts                   = 30/30/90/30/20
lane counts                        = 100/100
Train ↔ Dev lineage overlap        = 0
positive sample verify failures    = 0
Harness/oracle modifications       = 0
```

#### Held-out 的用途与物化

Held-out 只用于 Base vs SFT 最终盲测。物化结果只能包含：

- 冻结的公开 contract 与初始 parent source；
- 受保护的 Harness、Oracle 和 benchmark cases 引用；
- `task_id`、主 `task_type`、lane、lineage/implementation family 和环境 hash；
- Base/SFT 执行后产生的 apply/compile/correctness/performance receipts。

Held-out artifact 禁止包含 target patch、参考优化实现、训练 trajectory 或可被模型
检索的答案。Base 与 SFT 必须使用完全相同的 task、初始源码、Harness/Oracle、
生成预算、解码参数、GPU、ROCm、compiler 和 benchmark 重复次数。

GEAK 自带 benchmark 可以进入 Kernel Held-out，但必须同时满足：

1. 对应 source lineage、contract、shape cluster 和 Harness 未进入 Train/Dev；
2. 任务在训练前冻结，隐藏 correctness/benchmark case 不进入模型可见 prompt；
3. 不把 GEAK 当前最优实现或 benchmark reference 作为模型可见 seed；
4. 使用新 shape、边界条件或隐藏参数化 case，降低公开 benchmark 被预训练模型见过
   所造成的污染风险；
5. `source_suite=geak_native`，并单独报告 GEAK-native 子集指标；
6. 若同一 benchmark 曾用于生成 Train case，即使换语言或做轻微 shape 改写，也不得
   再作为 Held-out 独立任务。

Held-out task 从冻结开始不得用于：

- template 搜索语料；
- learned Skill、KB update 或 retrieval corpus；
- SFT/DPO/RL target；
- error-recovery 训练；
- prompt 示例；
- threshold、early stopping 或 checkpoint 调参。

当前 v4 split 是 lineage-level SHA256 确定性分配并继承 v3 assignment。现有容量为
6 个 Dev lineage、44 个 Dev generation requests，以及 12 个 Held-out lineage、
约 21 个 lane-task slots。Dev 可以通过同 split 的多 mode 真实采集达到 200 条；
Held-out 库距离 120 个独立 task 仍不足，禁止复制 task、改 shape 后冒充独立任务
或挪用 Train/Dev lineage 凑数。

物化 Held-out 120 前必须先完成以下硬门禁：

1. 扩充候选 catalog，使冻结的 Held-out 独立 task 达到 120；
2. 创建新的 append-safe split version，继承既有 v4 assignment，并重新计算
   split manifest 与 SHA256；
3. 实现独立 `eval-only` 生成/执行入口；该入口不得调用 SFT collector，也不得
   写出 patch target；
4. 按 `18/18/54/18/12` 固定主 task type，并按 Triton/HIP 各 60 个配平；
5. 生成受保护的 Held-out manifest 后，自动把 case ID、符号和 lineage 加入
   `protected_held_out_terms`；
6. 对 Train、Dev 和 Held-out 执行联合 lineage、contract、AST/diff 和文本泄漏
   审计，全部为 0 后才能开始 Base/SFT 最终评测。

在上述入口和容量门禁完成前，Held-out 状态必须报告为 `not_materialized`，不能将
现有 21 个 slot 宣称为 120 个任务，也不能生成带参考 patch 的
`processed/held_out.jsonl`。

未来完整 Phase 1/Phase 2 的规模目标保持不变：

```text
Phase 1：4 个组合 × 40–80 tasks = 160–320 held-out tasks
Phase 2：8 个组合 × 40–80 tasks = 320–640 held-out tasks
```

### 7.3 Split 审计

冻结后输出：

```text
splits/train_groups.jsonl
splits/dev_groups.jsonl
splits/held_out_groups.jsonl
```

每个文件计算 SHA256。任何修改都创建新 split version。

## 8. Pilot 数据生产

### 8.1 生成并验证任务

批量接口补齐后：

```bash
cd "$MULTITUNE_ROOT"
source .venv/bin/activate
multi-tune --config "$LUMEN_CODE_CONFIG" generate \
  --manifest cases/phase1-generation-requests.yaml \
  --output-catalog cases/phase1-cases.yaml \
  --stream
```

当前代码尚未实现该命令时，使用：

```bash
multi-tune --config "$LUMEN_CODE_CONFIG" interactive
```

并选择 `[2] generate kernel task`。这里只允许 pilot。

### 8.2 执行现有 case

单 case：

```bash
multi-tune --config "$LUMEN_CODE_CONFIG" run \
  --case <case-id> \
  --request "<frozen objective and constraints>" \
  --stream
```

整个 catalog：

```bash
multi-tune --config "$LUMEN_CODE_CONFIG" run --stream
```

省略 `--case` 会顺序执行 catalog 中全部 case。不要让多个外部进程绕过同一 GPU
lock 并发 benchmark。

### 8.3 每个 run 后的即时检查

检查：

```text
trajectory.jsonl
summary.json
sft_events.jsonl
sft_manifest.json
environment.json
round_N/plan.json
round_N/candidates.jsonl
```

拒绝 run 的情况：

- collector 文件缺失或 JSON 损坏；
- Git/config/environment SHA 缺失；
- parent source 不存在；
- direction 时间晚于 Engineer start；
- profile source hash 不匹配；
- independent verify 不存在；
- benchmark GPU 身份不明；
- Harness/oracle 被修改；
- GPU contention 或 timing variance 超阈值；
- hidden 内容进入模型 prompt。

### 8.4 Pilot Go/No-Go

只有同时满足以下条件才进入规模化生产：

- 四个 Phase 1 组合均有真实 ready worker；
- 每种 task type 至少有可验证样本；
- 100% 正样本有 parent source、patch 和 independent verify；
- 100% patch 可在 fresh workspace 应用；
- train/dev/held-out lineage leakage 为 0；
- Harness modification 为 0；
- 未知 GPU/ROCm/compiler provenance 为 0；
- timing unreliable 样本不会进入正样本；
- raw artifact 可通过 checksum 重建相同 processed JSONL。

## 9. Phase 1 规模化生产

### 9.1 调度单位

一个 job 绑定：

```text
dataset_version
worker_id
gpu_sku
language
architecture
task_id
task_type
generator_model_version
workflow_git_sha
geak_git_sha
attempt
```

Job queue 可以并发分配到多台机器，但每张 benchmark GPU 必须串行执行。

### 9.1.1 当前 8×gfx942 主机的 7-Worker 方案

当前主机采用以下固定拓扑：

```text
GPU 0：vLLM / generator model 服务
GPU 1：geak-worker-gfx942-1
GPU 2：geak-worker-gfx942-2
GPU 3：geak-worker-gfx942-3
GPU 4：geak-worker-gfx942-4
GPU 5：geak-worker-gfx942-5
GPU 6：geak-worker-gfx942-6
GPU 7：geak-worker-gfx942-7
```

可以同时保持 7 个端到端 case job in-flight，但必须分别控制三类并发：

```text
模型生成/Agent 请求并发：初始 3–4，压测稳定后最多 7
CPU/编译并发：           初始 2–4，避免容器、磁盘和编译缓存争用
GPU benchmark 并发：     最多 7；每张 GPU 同时严格为 1
```

不能把“7 张 benchmark GPU”解释为单个 case 同时跑 7 次。调度器必须优先分配
不同 `task_id`、不同 source lineage 或真实缺口；同一 case 的重复 attempt 只能用于
明确的噪声复测，不能增加 unique sample/lineage 计数。

#### Worker 配置

当前 CLI 的 `gpu_ids` 来自 YAML，不能让 7 个进程共享固定
`gpu_ids: "1"` 的 `configs/mi300x.yaml`。创建 7 个 worker 配置：

```text
configs/workers/gfx942-gpu1.yaml  gpu_ids: "1"
configs/workers/gfx942-gpu2.yaml  gpu_ids: "2"
configs/workers/gfx942-gpu3.yaml  gpu_ids: "3"
configs/workers/gfx942-gpu4.yaml  gpu_ids: "4"
configs/workers/gfx942-gpu5.yaml  gpu_ids: "5"
configs/workers/gfx942-gpu6.yaml  gpu_ids: "6"
configs/workers/gfx942-gpu7.yaml  gpu_ids: "7"
```

除 `gpu_ids` 和 `worker_id` 外，七份配置必须使用相同的：

- workflow、GEAK 和 AITER commit；
- model endpoint/version；
- container image/digest；
- timeout、baseline repeat 和 acceptance threshold；
- candidate/seed/split manifest version；
- SFT schema 与 dataset version。

每个 worker 启动时同时设置：

```bash
GPU_ID=<1..7>
export HIP_VISIBLE_DEVICES="$GPU_ID"
export GEAK_GPU_ALLOWED="$GPU_ID"
export GEAK_GPU_REQUIRE_IDLE=1
export GEAK_CONTAINER_NAME=geak-phase1-vllm
```

`gpu_ids`、`HIP_VISIBLE_DEVICES` 和 `GEAK_GPU_ALLOWED` 三者不一致时，job 必须在
baseline 前失败，不能自动回退到 GPU 1。

#### 7-Worker 调度状态机

```text
READY_CASE_QUEUE
→ acquire model-generation semaphore（初始容量 3–4）
→ TechLead/Engineer 生成 candidate
→ release model-generation semaphore
→ acquire worker-specific GPU lock
→ 确认该 GPU idle、architecture=gfx942、SKU/UUID 与 worker 注册一致
→ compile
→ correctness
→ performance
→ fresh-workspace independent verify（仍绑定同一 GPU）
→ release GPU lock
→ append result / rejection / provenance
→ 从 quota gap 领取下一个不同 case
```

一个 run 内有最多 3 个 Engineer candidate 时，这些候选可以并发请求 GPU 0 上的
模型服务，但在其绑定的 benchmark GPU 上仍须排队串行执行。7 个 run 同时进入
Engineer 阶段可能产生最多 21 个模型请求，因此初始必须使用全局模型 semaphore，
不能让 7 个 worker 无限制直连 vLLM。

#### 调度要求

1. queue item 必须绑定唯一 `worker_id + gpu_id + task_id + task_type + attempt`；
2. baseline、所有 candidate 和 independent verify 必须始终使用同一物理 GPU；
3. 每次性能测量前检查 foreign process、GPU busy、VRAM 和温度/功耗状态；
4. GPU 0 不进入 case benchmark，也不因暂时空闲而被 worker 抢占；
5. 允许共享只读 AITER/GEAK/Lumen checkout 和内容寻址 blob store；
6. 每个 run 使用独立 workspace、run ID、event log 和 result path；
7. benchmark cache 可共享只读内容，但编译输出和临时文件必须按 worker/run 隔离；
8. 同时满载导致 timing variance 超阈值时，自动把 benchmark 并发从 7 降到 4，
   然后降到 2；不能放宽噪声门禁；
9. 模型请求超时或 GPU lock 等待不算 kernel 失败，必须使用独立 rejection reason；
10. 每轮调度后报告 7 个 worker 的利用率、模型队列长度、GPU lock wait、
    compile/correctness/performance/verify pass rate 和 accepted sample/GPU-hour。

#### 分阶段启用

```text
Stage A：3 个 worker（GPU 1–3），模型并发 3
  运行至少 10 个不同 P0 cases，确认无 workspace/cache/GPU 串扰。

Stage B：4 个 worker（GPU 1–4），模型并发 3–4
  对比 Stage A 的 baseline variance 和 accepted sample/GPU-hour。

Stage C：7 个 worker（GPU 1–7），模型并发仍从 4 开始
  只有 timing unreliable rate、vLLM timeout 和编译失败率未恶化才提高模型并发。
```

进入 Stage C 的硬门禁：

- 至少 7 个不同且可运行的 validated cases；
- 每个 worker 配置通过启动检查；
- GPU lock 能区分 1–7；
- 并发 run 不共享可写 workspace；
- Stage B 的 timing unreliable rate 在允许范围内；
- vLLM GPU 0 无 OOM，模型请求延迟和超时率可接受；
- collector 在并发写入下保持完整，ETL 可按 run ID 重建。

### 9.2 配额控制

每天计算去重后的缺口，不按 raw run 数补量：

```text
quota_gap =
target_verified_unique_samples
- accepted_verified_unique_samples
```

优先补：

1. 缺少的 language-architecture 组合；
2. 缺少的 source lineage；
3. 缺少的 task type；
4. 缺少的 shape regime/dtype/format；
5. 有效 token 较少的组合。

禁止为了速度反复运行同一个容易成功的 kernel。

### 9.3 中断与恢复

MultiTune interactive 支持 `[c] continue`，会复制最新 candidate workspace、
保留 frozen baseline 并传递失败结果。恢复时：

- 使用新的 run ID；
- 保存 `parent_run_id`；
- 不重复计数已完成 candidate；
- 不伪造原始 model KV cache；
- 将恢复后的数据正确分类为 error-recovery 或 continuation provenance。

### 9.4 监控指标

每天报告：

- worker/GPU 健康；
- run 成功、失败、超时数；
- GPU lock 等待时间；
- compile/correctness/benchmark/verify pass rate；
- 每组合 verified unique samples；
- 每 task type 样本数；
- source lineage 数；
- 去重率；
- timing unreliable rate；
- GPU minutes per accepted sample；
- token 数与平均 prompt/output 长度；
- collector/ETL rejection reasons。

## 10. ETL

### 10.1 正样本选择

样本只有满足以下条件才能进入内部 staging 的 `processed/train.jsonl` 或
`processed/dev.jsonl`，随后才可打包进 HF `samples.jsonl`：

```text
patch_applies
AND compile_pass
AND correctness_pass
AND benchmark_valid
AND independent_verify_present
AND source_hash_matches
AND allowed_paths_only
AND no_harness_modification
AND no_leakage
```

对于优化正样本，还必须：

```text
verified_speedup > noise_threshold
AND all regression constraints pass
```

只修复 correctness 的 Error-recovery 样本可以不要求 speedup，但必须保证性能未超过
冻结的退化上限。

### 10.2 标准样本

```json
{
  "schema_version": "geak_kernel_sft_v1",
  "sample_id": "...",
  "task_type": "direction_conditioned",
  "split": "train",
  "input": {
    "contract": {},
    "parent_source": "...",
    "baseline": {},
    "profile": {},
    "direction": {},
    "error_feedback": null,
    "per_case_benchmark": null,
    "regression_constraints": null
  },
  "output": {
    "patch": "<unified diff>"
  },
  "labels": {
    "patch_applies": true,
    "compile_pass": true,
    "correctness_pass": true,
    "benchmark_valid": true,
    "verified_speedup": 1.18
  },
  "provenance": {
    "run_id": "...",
    "candidate_id": "...",
    "source_hash": "...",
    "patch_hash": "...",
    "lumen_git_sha": "...",
    "geak_git_sha": "...",
    "gpu": "gfx942",
    "gpu_sku": "MI308",
    "rocm_version": "...",
    "compiler_version": "...",
    "implementation_family_id": "...",
    "verify_source": "verify_engineer"
  }
}
```

训练时只对 Assistant 的 `output.patch` token 计算 loss。

### 10.3 Rejected 数据

失败样本写入 `processed/rejected.jsonl`，保留 reason：

```text
missing_parent_source
patch_does_not_apply
compile_failed
correctness_failed
benchmark_invalid
speedup_below_noise
regression_failed
verify_missing
harness_modified
source_hash_mismatch
profile_source_mismatch
split_leakage
timing_unreliable
unsupported_contract
collector_incomplete
```

Failed patch 不能作为正向 SFT target。它可以用于未来 preference 数据，或作为
Error-recovery 输入的一部分。

### 10.4 去重

至少执行：

- exact source hash；
- exact patch hash；
- normalized text hash；
- AST/IR 相似性；
- diff hunk 相似性；
- source lineage；
- shape cluster；
- 跨语言 implementation family。

若同一 patch 多次运行，保留最可靠的一份训练样本，其余 measurement 可合并到
provenance，不增加样本计数。

## 11. Independent Re-verification

ETL 后对所有正样本或至少按风险分层的完整抽样重新验证：

1. 创建 fresh workspace；
2. 加载保存的 parent source；
3. 应用 patch；
4. 验证只有 allowed paths 改变；
5. 使用冻结 Harness；
6. 运行 compile；
7. 运行公开与隐藏 correctness；
8. correctness 通过后运行 benchmark；
9. 保存新的 receipt；
10. 比较原始 verify 与 re-verify。

以下情况必须 quarantine：

- patch 无法复现；
- 正确性不稳定；
- speedup 落入噪声；
- kernel 未实际 launch；
- 输入被非法修改；
- 检测到 reference delegation；
- 原始与重验 GPU/software identity 不匹配且无法解释。

## 12. General Coding Replay、Dev 与 Held-out

Kernel SFT 之外必须维护独立 General Coding 数据域，用于防止模型只会写
HIP/Triton kernel、却损失普通代码理解、修改和测试修复能力。该数据域分为三部分：

```text
General Coding Replay Train   Phase 1: 500；Phase 2: 700–1,200
General Coding Dev            Phase 1: 80 个 verified tasks
General Coding Held-out       官方 split/date window + 40 个 Multi-SWE-bench eval tasks
```

### 12.1 Replay Train

Phase 1 固定目标为 500 条，不再使用 `400–600` 作为执行时的浮动口径。500 条与
当前 2,000 条 Kernel Train 合并后占 20% 的行数；训练 sampler 仍必须把
General Coding 的 Assistant loss token 占比限制在 15%–20%。

最终 500 条按任务类型配额：

```text
repository-level bug fix / test repair    220
function implementation                   100
refactor / API adaptation                  80
build / dependency / config                50
non-kernel performance repair              30
typing / validation / docs-with-test       20
Total                                     500
```

按来源设置 selection target：

```text
SWE-Gym/SWE-Gym                                             250
PrimeIntellect/Multi-SWE-RL-Verified                        250
Total                                                       500
```

Phase 1 必须把两个来源都锁定到 immutable dataset commit；SWE-Gym 固定为
`bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`，Multi-SWE-RL-Verified 固定为
`80de95c62ac792c99dcfa8e26569bcd7d036bdc3`。固定来源目标不表示这 500 条已经
materialize 或 verify；每条仍须通过下述 repository replay、license、去重和 fresh
workspace 门禁。

按主要实现语言设置 selection target：

```text
Python                                    250
C/C++                                      75
JavaScript/TypeScript                      75
Go                                         50
Rust                                       50
Total                                     500
```

Phase 1 的 source × language 交叉配额固定为：SWE-Gym 提供 Python 250；
Multi-SWE-RL-Verified 提供 C/C++ 75、JavaScript/TypeScript 75、Go 50、Rust 50。
Multi-SWE-RL-Verified 的 Java rows 不计入这 500 条，且不得用 Java 改写或翻译补其他
语言格子。六类 task type 在两个来源内联合配平，但总数仍必须精确为
`220/100/80/50/30/20`。

三组配额是正交维度，同一条样本同时占一个 task type、一个 source 和一个 language
格子，不能相加成 1,500 条。任一交叉格子缺少可验证候选时必须报告 deficit。语言和
task type 总额必须保持上述 exact quotas；来源调整只能通过书面批准的
`quota-overrides.yaml`，不得降低测试门禁、复制样本或翻译同一 patch 补格子。

以下来源决策属于 Phase 1 冻结策略：

- `nvidia/OpenCodeInstruct` 不进入 Replay。它虽包含 tests/执行状态字段，但记录不能
  可靠恢复对应 repository parent、精确 base commit 和 target diff，因而不能执行
  “checkout parent → apply target patch → replay tests”的正样本审计；
- `bigcode/commitpackft`（CommitPackFT）不是 primary source；commit/diff 记录本身
  不等于数据集原生、execution-backed 的 targeted/full-test 证据；
- `SWE-bench/SWE-smith` 只作为 mutation replay fallback。只有 primary source 经
  全量 inventory/verify 后仍有 source deficit，才可按批准的 quota override 使用；
  必须从冻结 clean revision 重放 mutation，证明 mutation 使既有测试失败、repair
  恢复测试，并保存 mutation/repair 双 patch、测试 receipt 和 fresh replay。不得把
  合成 mutation 与真实 upstream issue 混称。

#### 12.1.1 候选池

分别建立足量候选池，再通过执行门禁筛到固定的 `250 + 250 = 500`：

```text
SWE-Gym candidates                               >= 750
Multi-SWE-RL-Verified candidates                 >= 750
SWE-smith mutation candidates                    仅在批准 fallback 后建立
```

候选 inventory 必须先记录 dataset revision、license、原始 repository、base
commit、problem ID、language、预计 task type 和测试入口。候选数不是训练样本数，
未执行验证前不得计入 500。

#### 12.1.2 收集与验证状态机

```text
FREEZE_SOURCE_REVISIONS
→ FREEZE_EXCLUSION_REGISTRY
→ FREEZE_SOURCE_LINEAGE_MAP
→ BUILD_CANDIDATE_INVENTORY
→ MATERIALIZE_PARENT_WORKSPACE
→ RUN_PARENT_TESTS
→ APPLY_TARGET_PATCH
→ RUN_TARGETED_TESTS
→ RUN_FULL_REGRESSION_TESTS
→ NORMALIZE_PATCH_AND_PROVENANCE
→ FRESH_WORKSPACE_REPLAY
→ CROSS_SOURCE_LINEAGE_AND_EVAL_AUDIT
→ DEDUP_AND_ASSIGN_TRAIN
→ QUOTA_SELECT_500
→ APPLY_AND_AUDIT_QUOTA_OVERRIDES
→ PACKAGE_HF_ARTIFACTS
→ QUALITY_LEAKAGE_COVERAGE_CHECKSUM
```

`FREEZE_EXCLUSION_REGISTRY` 至少包含 HumanEval、MBPP Sanitized、冻结的
LiveCodeBench 日期窗口、Coding Dev problem IDs、40 个 Multi-SWE-bench eval-only
task IDs、Kernel Held-out lineage/符号和所有已知 benchmark mirror。排除清单必须按
revision 保存并参与 manifest checksum。

`FREEZE_SOURCE_LINEAGE_MAP` 必须把 dataset row 映射到原始 upstream repository、
problem/PR、base commit、target commit/patch、测试 patch/hash 和 normalized diff
lineage。由于 SWE-Gym、Multi-SWE-RL-Verified、Multi-SWE-bench 与 SWE-smith 可能
共享 repository 或衍生问题，只按 dataset ID 去重不合格；跨来源 lineage、patch 或
测试集合碰撞必须整体拒绝或固定到同一非训练 split。

每条候选必须在隔离 workspace 中执行：

1. checkout 精确 base commit，恢复模型可见 parent source、依赖 lock 和 tests；
2. 在 parent 状态运行 targeted tests，确认任务不是已解决；若任务语义允许部分
   baseline tests 通过，必须记录预期失败集合；
3. 应用原始 target patch，确认只修改 allow-list path；
4. 运行 compile/typecheck、targeted tests 和 full regression tests；
5. 保存 stdout/stderr、exit code、duration、container digest 和 test selection；
6. 重新从干净 workspace replay 一次，结果一致才可进入正样本池。

仅当以下全部满足时接收：

- 保存原始 dataset ID、revision、license 和 test status；
- 使用 `sample_domain=general_coding` 和独立
  `task_type=general_coding_replay`，并用 `coding_task_type` 保存六类配额中的
  primary task type；
- parent source、base commit、target patch 和测试可完整恢复；
- patch 在 fresh workspace 可应用，且只修改 allow-list path；
- compile/typecheck、targeted tests 和 full regression tests 全部通过；
- 与 Kernel Train/Dev/Held-out、Coding Dev/Held-out 和通用 benchmark 去重；
- 不能把解释文本、测试答案或 benchmark reference 伪装成 patch target；
- HumanEval、MBPP、LiveCodeBench 等最终评测题禁止进入 Replay Train。

`quota-overrides.yaml` 每条 override 必须记录原 source target、new source、数量、
触发 deficit、候选/拒绝统计、批准人、批准时间、source revision 和 fallback
lineage policy。它只能改变来源边际；`Total=500`、六类 task-type exact quota、五类
language exact quota、测试/去重/泄漏门禁不得 override。无 override 时文件也必须以
空列表参与 checksum。

标准 rejection reason 至少包括：

```text
license_unknown
license_incompatible
parent_revision_missing
parent_not_reproducible
target_patch_missing
patch_apply_failed
tests_missing
dependency_restore_failed
parent_already_passes
targeted_tests_failed
regression_tests_failed
test_timeout
fresh_replay_mismatch
forbidden_path_modified
benchmark_overlap
split_leakage
duplicate_source
duplicate_patch
```

#### 12.1.3 去重、多样性与 token 配比

至少按 repository、problem ID、source hash、normalized patch hash、AST、diff
hunk 和测试集合联合去重。额外限制：

```text
单一 upstream repository               <= 15 条
同一 source lineage                    <= 2 条
同一 normalized patch                   = 1 条
单一数据来源占比                         <= 60%
```

语言、来源和 task type 都必须在去重后按 quota selection，不能先配平再去重。若一条
样本满足多个 task type，只按其冻结的 primary task type 计数。

训练混合按 tokenizer 后的 Assistant loss token 而不是 JSON 行数控制：

```text
General Coding loss tokens / all loss tokens = 15%–20%
```

不得截断到破坏 patch、测试语义或 Assistant EOT；超长样本应进入独立长度 bucket，
通过 length-aware sampler 降权。每个 epoch 同时报告 Kernel/General Coding 的行数、
有效 loss token、采样权重和实际 token 占比。

### 12.2 General Coding Dev

Phase 1 固定 80 个任务，推荐覆盖：

```text
repository-level bug fix / test repair    40
function-level implementation             20
refactor / API adaptation                 20
```

Coding Dev 必须与 Replay Train 按 repository、problem ID、source lineage 和
normalized patch 去重，且具有可执行测试。它只用于训练期 retention 监控：

- Kernel Dev 执行指标仍是 checkpoint selection 主依据；
- Coding Dev pass@1 是硬退化门禁；
- Coding Dev 不参与梯度更新；
- 若候选 checkpoint 的 Coding Dev pass@1 相对 Base 下降超过 2 个百分点，不能选为
  最终 checkpoint，即使 Kernel Dev 更高。

### 12.3 General Coding Held-out

Coding Held-out 与 120 个 Kernel Held-out tasks 分开计数、分开存储、分开报告。
至少冻结以下评测来源及其精确 revision/config：

- HumanEval 完整 official test split；
- MBPP Sanitized 完整 official test split；
- LiveCodeBench 固定日期窗口；
- 40 个 eval-only repository tasks，来自
  `ByteDance-Seed/Multi-SWE-bench@56ff018c04a38e27ada1e9d0a6d5839a51f88f0d`。

这 40 个任务替代当前不可获得的 private-repository hidden-test inventory，但不是
训练数据，也不能称为私有或训练前未知。选择清单必须在训练前冻结，并同时满足：

1. 与 Replay Train、Coding Dev、Kernel 数据按 upstream repository、problem/issue/PR
   ID、source lineage、base/target commit、normalized solution patch、test patch、
   AST/diff hunk 全部 disjoint；
2. 精确记录 dataset revision、task ID、upstream repository、base commit、container
   digest 和 harness revision；任何 floating branch、`latest` 或运行时重新选题都拒绝；
3. solution patch、test patch、oracle/expected fail-to-pass 与 pass-to-pass 集合只存于
   访问隔离的评测服务或 private eval repository，不进入 Train/Dev repository、
   prompt、Skill、KB、retrieval corpus、checkpoint selection 或模型可见 workspace；
4. 模型可见 task manifest 只能引用上述 protected artifacts 的 SHA256、公开 issue、
   精确 base revision 和初始源码；不得出现可逆路径、patch 内容、测试答案或 oracle；
5. private/eval repository 使用与训练数据不同的 credentials、artifact namespace、
   logs 和 retention policy；Base/SFT 只提交 candidate patch，由评测服务应用 protected
   tests/oracle 并产生 receipt；
6. 冻结后生成 repository/problem/lineage/patch/test-set 联合泄漏审计，所有 overlap
   必须为 0，且 40 个任务逐项确认 execution environment 可重建后才可标记
   `materialized`。在此之前状态为 `planned_not_verified`。

最终选择 checkpoint 后才能运行 Coding Held-out。它不得用于 threshold 调整、
early stopping、prompt/Skill 选择或多 checkpoint 试跑。对已公开 benchmark，
必须报告“可能已存在于 base-model pretraining”的限制，不能把结果宣称为严格
训练前未知。Multi-SWE-bench 同样是公开任务，可能已出现在 base-model pretraining；
protected solution/tests 只防止本项目流水线直接泄漏，不能消除预训练污染。

HumanEval、MBPP Sanitized 和冻结的 LiveCodeBench 规则保持不变：完整 official
split/date window 均为 held-out，禁止训练、retrieval、few-shot、Skill/KB 更新和
checkpoint selection。

General Coding 报告至少包含：

```text
compile/test pass@1
patch apply rate（repository-level）
problem-level pass@1
各 suite macro average
Base → SFT 绝对变化
95% paired bootstrap CI
```

Kernel 与 General Coding 的最终门禁分别计算，再汇总为 retention gate：

```text
Kernel valid optimization 显著改善
AND Kernel correctness 不下降超过 2 percentage points
AND General Coding macro pass@1 不下降超过 2 percentage points
```

## 13. 质量与泄漏验收

运行：

```bash
python scripts/validate_sft_dataset.py \
  --manifest "$STAGING_DATASET_ROOT/dataset_manifest.json"

python scripts/audit_sft_leakage.py \
  --manifest "$STAGING_DATASET_ROOT/dataset_manifest.json"

python scripts/report_sft_coverage.py \
  --manifest "$STAGING_DATASET_ROOT/dataset_manifest.json"
```

这些是第 5 节要求实现的目标接口；在脚本存在前不得跳过等价检查。
三个 staging audit 通过后，还必须验证 HF 打包结果中的 `manifest.json`、
`samples.jsonl`、所有相对 artifact path 和 checksum；不能只验证转换前数据。

发布门禁：

```text
JSON/schema errors                         = 0
missing required provenance               = 0
positive samples without independent verify = 0
patch apply failures in positive samples  = 0
Harness/oracle modifications              = 0
train/dev/held-out lineage overlap        = 0
held-out prompt/Skill/KB leakage           = 0
General Coding train/eval problem overlap  = 0
Kernel ↔ General Coding lineage overlap    = 0
unknown language/architecture             = 0
unsupported native quantization accepted  = 0
duplicate samples counted toward quota    = 0
```

Coverage report 必须按以下维度分别列出：

- language；
- architecture；
- exact GPU SKU；
- task type；
- operator；
- dtype/format；
- shape regime；
- source lineage；
- implementation family；
- split；
- valid Assistant token。

## 14. 数据版本冻结

冻结前：

1. 停止新 job；
2. 等待 active GPU verification 完成；
3. 标记中断 run；
4. 固定 raw run list；
5. 重跑 ETL、validation、leakage 和 coverage；
6. 将 staging 数据打包为第 1 节定义的 HF repository layout；
7. 验证 `samples.jsonl` 可加载、split 数量正确且所有相对 path 可解析；
8. 计算 HF repository 内所有文件 checksum；
9. 同步写 `README.md` 与 `dataset_card.md`；
10. 将 `manifest.json` 标记为 immutable。

生成 checksum：

```bash
cd "$HF_DATASET_ROOT"
find . -type f ! -name checksums.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum > checksums.sha256
sha256sum -c checksums.sha256
```

Dataset card 必须说明：

- 目的与禁止用途；
- 数据版本和创建日期；
- Lumen/GEAK/model/container SHA；
- 硬件与软件环境；
- 数据数量与 token；
- 语言/架构/task type 分布；
- task 与 source lineage 数；
- 筛选、验证、去重和 split 方法；
- known limitations；
- license/NOTICE；
- held-out 保护策略；
- rejection 数量和原因；
- benchmark 噪声与可靠性标准。

## 15. Phase 1 完成条件

Phase 1 只有满足以下条件才完成：

- 3,000–4,000 条 verified、unique Kernel SFT 正样本；
- 四个组合各 750–1,000 条，或有书面批准的偏差；
- 每组合 40–80 个独立 held-out tasks；
- 五种 task type 均有覆盖；
- 至少 10%–20% 合格通用 Coding replay；
- General Coding Dev 80 个任务已冻结且不参与训练；
- General Coding Held-out suite/revision/date window 已冻结且未用于 checkpoint 选择；
- Kernel 与 General Coding 分项报告完成，retention gate 通过；
- 全部发布门禁通过；
- Dataset 可从 raw artifact 可重复构建；
- Base vs SFT evaluation 已冻结但 held-out 未泄漏。

## 16. Phase 2 执行

Phase 1 冻结后：

1. 为 FlyDSL/Gluon 安装并锁定依赖版本；
2. 分别在 gfx942/gfx950 做真实 smoke；
3. 更新 support matrix；
4. 为新增组合创建独立 task 与 split groups；
5. 确认与 Phase 1 held-out 不形成翻译/lineage 泄漏；
6. 新增约 3,000–4,000 条 verified unique samples；
7. 重采样 Triton/HIP，避免旧语言或新语言按 token 数垄断；
8. 将总量扩展到 6,000–8,000；
9. 将通用 replay 扩到 700–1,200；
10. 重新执行全量验证、泄漏审计和数据冻结。

必须保留 Phase 1 dataset/checkpoint，以评估加入 FlyDSL/Gluon 后 Triton/HIP 是否
退化。

## 17. Agent 状态机

Agent 按以下状态执行，不得跳级：

```text
BOOTSTRAP
→ REPO_PINNED
→ ENV_VERIFIED
→ COLLECTOR_READY
→ SUPPORT_MATRIX_READY
→ SPLITS_FROZEN
→ PILOT_RUNNING
→ PILOT_ACCEPTED
→ PHASE1_COLLECTING
→ PHASE1_ETL
→ PHASE1_FROZEN
→ PHASE2_PREPARED
→ PHASE2_COLLECTING
→ PHASE2_ETL
→ DATASET_FROZEN
```

每次状态转换写入 append-only operation log，包括：

```json
{
  "from": "ENV_VERIFIED",
  "to": "COLLECTOR_READY",
  "timestamp": "...",
  "actor": "...",
  "evidence": ["..."],
  "checks_passed": ["..."],
  "git_shas": {},
  "config_hash": "..."
}
```

## 18. 必须停止并报告的情况

遇到以下情况，Agent 必须停止当前生产队列，不得自行绕过：

- branch 或 Git SHA 与冻结 manifest 不一致；
- Hardware architecture 与 task contract 不一致；
- native MXFP 请求落到 gfx942；
- GPU identity 无法确定；
- Harness/oracle/test 被修改；
- independent verify 不可用；
- collector 丢事件、文件损坏或 source 不可恢复；
- split 泄漏；
- benchmark GPU contention 持续存在；
- 正确性或 timing 显著不稳定；
- 模型 endpoint/chat template 未经记录发生变化；
- secret 出现在 raw artifact；
- FlyDSL/Gluon 组合没有可信执行链路；
- 需要删除或覆盖 raw artifact；
- 任何必须字段只能通过猜测补齐。

报告必须包含：

```text
current state
affected run/task/sample IDs
observed evidence
scope of affected data
whether data must be quarantined
safest next action
```

## 19. SFT-f 统一数据集（2026-10-06 更新）

基于 from-scratch benchmark 的发现（见 `agent-loop-efficiency-benchmark.md`），我们对 SFT 数据集进行了增强和格式统一，用于下一轮训练（SFT-f）。

### 19.1 数据集位置

- HuggingFace: `Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP`
- 本地: `/home/danyzhan/geak_sft_dataset/merged_sft_v2/train.jsonl`
- 格式转换脚本: `/home/danyzhan/.claude/jobs/95572d14/tmp/`
  - `gen_complete_impl_data.py` — 从 wave-2000 patches 生成 complete implementation 样本
  - `unify_format.py` — 统一 geak_kernel_sft_v1 → messages 格式
  - `convert_all.py` — 批量转换并上传 HF

### 19.2 统一格式

所有文件统一为 `messages` chat 格式（兼容 Qwen3-Coder SFT 训练）：

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert GPU kernel engineer..."},
    {"role": "user", "content": "<task description + kernel code>"},
    {"role": "assistant", "content": "<diff patch or complete kernel>"}
  ],
  "task_type": "direction_conditioned|cold_start|error_recovery|profile_guided|regression_balance|complete_implementation",
  "operator": "gemm|rms_norm|mha|...",
  "data_source": "patch_v1|complete_impl_v1",
  "schema_version": "messages_v1"
}
```

### 19.3 数据组成

| 文件 | 样本数 | 内容 |
|------|--------|------|
| train.jsonl | 3164 | 2000 patch + 1164 complete implementation |
| dev.jsonl | 200 | Patch 优化（从 geak_kernel_sft_v1 转换） |
| samples.jsonl | 2700 | 完整样本池（从 geak_kernel_sft_v1 转换） |

### 19.4 Task Type 分布（train.jsonl）

| 类型 | 数量 | 输入 | 输出 | 训练目标 |
|------|------|------|------|----------|
| direction_conditioned | 907 | kernel + 优化方向 | diff patch | 定向优化 |
| complete_implementation | 1164 | 仅 operator spec（无代码） | 完整 kernel | 从零写 kernel |
| cold_start | 307 | kernel，无方向 | diff patch | 自主优化 |
| error_recovery | 293 | 损坏的 kernel + 错误 | fix patch | 调试修复 |
| profile_guided | 293 | kernel + profiling 数据 | 优化 patch | 性能调优 |
| regression_balance | 200 | kernel + 约束 | balanced patch | 安全优化 |

### 19.5 数据来源

- **patch_v1**（2000）: 来自 `phase1-production-wave-2000-v1` agent runs
- **complete_impl_v2**（1164）: 来自 `phase1-production-wave-2000-v1`，通过 fuzzy patch apply 到 parent_source 得到最终优化 kernel，覆盖 23 种 operator

### 19.6 与 Benchmark 无重叠

- 训练数据: `phase1-production-wave-*` task pools（23 operators）
- Benchmark: `held-out-benchmark-aiter`（8 operators, 100 tasks）
- 验证: 0 内容重叠

### 19.7 相比上一版变更

1. **完整 wave-2000 patch 数据**（2000）: 包含全部 2000 条 verified patch 样本（之前仅 300 条 wave-300）
2. **新增 complete_implementation 样本**（1164）: 解决 benchmark 发现的核心问题 — SFT 模型无法从零写 kernel。使用 robust fuzzy patch apply 最大化 wave-2000 的利用率（从 129 扩展到 1164）
3. **统一格式**: 所有数据从 `geak_kernel_sft_v1` 结构化 JSON 转换为 `messages` chat 格式
4. **覆盖 23 种 operator**: gemm, mha, mla, paged_attention, rms_norm, rope_kv_cache, sampling, scaled_quant_gemm, fused_silu_mul, blockscale_gemm 等

## 20. 当前已知事实与缺口

执行本 runbook 时不要误判当前能力：

- MultiTune 已支持 existing case、interactive task generation、普通非交互 run、
  GPU lock、独立 verify 和 trajectory；
- 当前普通 run 会生成 `trajectory.jsonl` 与 `summary.json`；
- `cases/generation_requests.yaml` 是 manifest 示例；
- `multi-tune generate --manifest ...` 当前尚未实现；
- `sft_events.jsonl`、`sft_manifest.json`、结构化 plan/candidate sidecar 和离线
  SFT ETL 当前尚未完成；
- Direction-conditioned 是当前主要路径，其他 task type 需要显式采集模式；
- gfx942 FP8 A8W8 已有验证记录；
- gfx950 native MXFP 必须在真实 MI350/MI355 上重新验证；
- GEAK 不负责安装每种 kernel 的 PyTorch/Triton/FlyDSL/HIP toolchain，必须先让
  baseline 在目标环境中独立 build/run。

因此，正确顺序是：

```text
先补齐 collector、批量接口、task mode 和 ETL
→ 再做小规模 pilot
→ pilot 通过后才开始 3,000–4,000 条 Phase 1 生产
→ Phase 1 冻结后再扩展 FlyDSL/Gluon
```
