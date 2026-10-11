# GEAK Kernel Coding SFT Plan

## 目标与范围

第一版模型定位为单轮 **Kernel Coding Policy**：

- 输入当前可见的任务契约、源码和执行反馈；
- 一次生成可应用的 `unified diff`；
- 不负责候选选择、停止、晋升或最终验证；
- Skill 提供优化知识，Harness 提供真实反馈，Orchestrator 负责控制决策。

目标硬件按两个架构家族组织：

- gfx942 family：MI300/MI300X/MI308；
- gfx950 family：MI350/MI350X/MI355。

训练分两步扩展语言：

1. 第一阶段：Triton、HIP × gfx942、gfx950，共 4 个组合；
2. 第二阶段：加入 FlyDSL、Gluon，形成 4 种语言 × 2 种架构，共 8 个组合。

每次测量必须记录准确 GPU SKU。相同 `gfx` 家族可以共享代码模式，但 MI308
的 latency 不能标记为 MI300 latency，MI355 的 latency 也不能直接标记为
MI350 latency。

## 1. 统一训练数据格式

五类样本使用统一 schema：

```json
{
  "schema_version": "geak_kernel_sft_v1",
  "sample_id": "run-round-engineer",
  "task_type": "cold_start",
  "split": "train",
  "input": {
    "contract": {
      "operator": "gemm",
      "architecture": "gfx942",
      "m": 1,
      "n": 2048,
      "k": 4096,
      "input_dtype": "fp16",
      "output_dtype": "fp16",
      "layout": "A[M,K] @ B[K,N]",
      "language": "triton",
      "language_version": "...",
      "backend_version": "..."
    },
    "parent_source": "...",
    "baseline": {},
    "profile": null,
    "direction": null,
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
    "source_hash": "...",
    "patch_hash": "...",
    "gpu": "gfx942",
    "gpu_sku": "MI308",
    "rocm_version": "...",
    "compiler_version": "...",
    "implementation_family_id": "...",
    "verify_source": "verify_engineer"
  }
}
```

训练主目标使用 `output.patch` 对应的 patch token 加 Assistant EOT token 的
token-level causal cross-entropy。Prompt、Assistant role header、Harness 输出、
性能数据和验证标签不参与 loss；EOT 参与 loss，用于训练模型在 patch 结束后停止生成。

训练与 checkpoint 选择遵循：

- 采样时按 lane、task type 和 implementation family 配平，不按 speedup 粗暴加权；
- `eval_loss` 只用于监控过拟合和训练稳定性；
- checkpoint 选择以 Dev 上的 patch apply、compile、correctness 和 benchmark 为主，
  `eval_loss` 为辅。

数据构造必须满足：

1. `parent_source` 是生成 patch 时模型真实看到的源码；
2. patch 必须能够应用到该源码；
3. 输入中只能包含模型当时真实可见的信息；
4. 不能删除历史样本中的 profile/direction 后将其伪装成 cold-start；
5. speedup 只能来自独立验证，不能使用 Engineer 自报结果。

## 2. 构造五类 Coding 数据

### 2.1 Cold-start

```text
contract + 原始源码 + baseline
→ verified patch
```

用于训练模型第一次看到 kernel 时直接生成优化代码。

这类数据需要专门运行不提供 profile 和 TechLead direction 的任务。不能从使用过这些信息的历史 trajectory 中事后删除字段。

### 2.2 Profile-guided

```text
contract + 原始源码 + profile
→ verified patch
```

用于训练模型根据 launch、memory、compute、occupancy 等瓶颈生成代码。

Profile 必须针对 `parent_source` 和同一运行环境产生，不能将修改后的 profile 配给修改前源码。

### 2.3 Direction-conditioned

```text
contract + 原始源码 + profile + TechLead direction
→ verified patch
```

用于训练 Engineer 执行外部系统给出的明确优化方向。

Direction 必须在 Engineer 执行前持久化为类似 `round_N/plan.json` 的结构化记录。不能使用事后的 `worker_result.strategy` 反推输入。

### 2.4 Error-recovery

```text
失败源码 + 精确 compile/correctness 错误
→ 一次修复成功的 patch
```

修复 patch 必须基于失败后的源码，并满足：

- patch 可以干净应用；
- compile 通过；
- 完整 correctness 通过；
- 没有破坏其他 case；
- 由 `verify_engineer` 独立复验。

失败 patch 可以作为输入或后续 preference/rejected 数据，不能作为正向 SFT target。

### 2.5 Regression-balance

首先发现以下候选：

```text
aggregate speedup > 1
但至少一个关键 case 超出 regression budget
```

再构造修复样本：

```text
当前源码
+ per-case benchmark
+ “整体变快但关键 shape 严重退化”的反馈
+ regression constraints
→ balanced verified patch
```

正样本必须满足：

- aggregate performance 仍有有效提升；
- 每个关键 case 均满足退化限制；
- correctness 和 benchmark 有效；
- 独立 verify 通过。

## 3. 数据配比、采集与切分

单个 language-architecture 组合的 1,000 条正样本可从以下比例开始：

```text
Cold-start             150
Profile-guided         150
Direction-conditioned  450
Error-recovery         150
Regression-balance     100
```

当前 gfx942 双 lane 的 Dev 与 Held-out 规划为：

```text
                         Cold  Profile  Direction  Error  Regression  Total
Dev verified samples       30       30         90     30          20    200
Held-out tasks             18       18         54     18          12    120
```

两者均保持 `15% / 15% / 45% / 15% / 10%` 的 task type 配比。Profile-guided、
Direction-conditioned、Error-recovery 和 Regression-balance 的上下文必须专门从
对应 split 的冻结任务产生，不能从 train trajectory 删除或改写字段后伪造。每个
Held-out task 只计入一种主 task type，避免重复计数。Held-out 的代码、patch、
隐藏 Harness/Oracle、case ID 和衍生 Skill 不得进入训练或 retrieval。

当前 MultiTune 可见轨迹中，17/17 条 Engineer `agent_loop` 都携带
`direction_id`，且没有发现 `fallback_*` direction。因此现有生成路径实际上
接近 100% direction-conditioned，上述配比提高了这类样本占比，同时保留
cold-start、profile-guided 和恢复能力。

采集时必须额外记录 direction 来源：

```json
{
  "direction_source": "tech_lead",
  "direction_id": "r1_d0",
  "direction_prompt": "...",
  "direction_prompt_hash": "...",
  "direction_created_at": "...",
  "fallback_reason": null,
  "skill_refs": [],
  "skill_corpus_hash": "..."
}
```

`direction_source` 的允许值至少包括
`tech_lead | fallback | skill | user`。方向必须在 Engineer 执行前持久化，
否则不能构造 direction-conditioned 样本。

### 1,000 条样本的适用边界

对于 `Qwen3-Coder-30B-A3B + LoRA/adapter`，1,000 条高质量 Kernel 样本：

- 足够验证数据管线、输出格式和 domain alignment；
- 足够观察 Patch Apply、Compile Pass@1 和 Correctness Pass@1 是否改善；
- 可以作为第一次 held-out 对照实验；
- 不足以证明广泛 GEMM 泛化，也不适合直接做全参数 SFT。

这 1,000 条必须满足：

1. 来自至少 100–200 个独立 source lineage，而不是同一 kernel 的相邻 patch；
2. gfx942/gfx950 分布平衡，记录准确 MI300/MI308/MI350/MI355 SKU，并分别出评测报告；
3. 覆盖计划中的 dtype/format 和 decode、small-batch、prefill 等 shape cluster；
4. 完成文本、AST、diff 和 source-lineage 去重；
5. 训练样本之外另设 100–150 个独立 held-out tasks；
6. 每条正样本都有 independent verify，不能仅有 Engineer 自报结果；
7. 统计有效训练 token、独立 task 数和去重后样本数，不能只报告 JSON 行数。

单组合 POC 建议训练 1–2 epoch；若 held-out 指标有效提升，再进入下面的
3,000–4,000 条阶段一多语言训练。

### 两阶段 Language-Architecture 扩展

训练单元由 `(language, architecture, contract family, task_type)` 定义。必须先
建立真实支持矩阵；无法在目标硬件上 compile/correctness/benchmark 的组合不得
为了配平而制造样本。

#### 阶段一：Triton、HIP × gfx942、gfx950

四个核心组合：

```text
Triton × gfx942
Triton × gfx950
HIP    × gfx942
HIP    × gfx950
```

阶段一建议数据量：

```text
Verified Kernel SFT：      3,000–4,000 条
每个组合：                   750–1,000 条
通用 Coding replay：         400–600 条
Held-out tasks：             每个组合 40–80 个，总计 160–320 个
```

每个组合内部继续使用约 `15% / 15% / 45% / 15% / 10%` 的
Cold-start、Profile-guided、Direction-conditioned、Error-recovery、
Regression-balance 配比。某类样本不足时必须报告真实数量，不能复制样本凑比例。

阶段一目标是证明：

- Triton 与 HIP 的 Patch Apply/Compile/Correctness Pass@1 均提升；
- gfx942 与 gfx950 分平台不退化；
- 统一 adapter 不明显弱于单语言 adapter；
- 通用 Coding benchmark 通过能力保持门禁。

#### 阶段二：加入 FlyDSL、Gluon，扩展到八个组合

新增：

```text
FlyDSL × gfx942
FlyDSL × gfx950
Gluon   × gfx942
Gluon   × gfx950
```

阶段二数据量指整个八组合数据集的最终规模：

```text
Verified Kernel SFT 总量： 6,000–8,000 条
每个有效组合：               750–1,000 条
阶段二新增数据：             约 3,000–4,000 条
通用 Coding replay 总量：     700–1,200 条
Held-out tasks：             每个组合 40–80 个，总计 320–640 个
```

阶段二不能只追加 FlyDSL/Gluon 后直接继续训练，还应：

1. 重采样 Triton/HIP，避免新语言或旧语言在 token 数上占据绝对优势；
2. 使用 `language`、`language_version`、`backend_version` 和
   `compiler_version` 明确条件化；
3. 将同一数学 GEMM 的多语言实现关联到同一 `implementation_family_id`；
4. 防止同一实现的语言翻译版本跨 train/held-out 泄漏；
5. 分别报告 8 个组合的指标，并计算各组合等权的 macro average；
6. 保留阶段一 checkpoint，以判断加入 FlyDSL/Gluon 是否导致 Triton/HIP 退化。

若 FlyDSL 或 Gluon 在某个 architecture 上没有真实、稳定的执行链路，该组合应
标记为 `unsupported` 或 `not_ready`，不能用另一个架构的结果替代。

### GEAK 采集

在不改变 GEAK 评测语义的前提下增加 append-only sidecar：

1. 在 `kernel_lane.js` 的 `agentT()` 外层记录 prompt、role、phase 和 structured response 到 `EVAL_DIR/sft_events.jsonl`；
2. Engineer 执行前，将允许修改的源码保存到内容寻址存储，并记录 `source_hash`；
3. 独立 verify 后，将 direction、source hash、patch hash 和 verify 结果写入 `round_N/candidates.jsonl`；
4. Director 完成后写入 `EVAL_DIR/sft_manifest.json`；
5. artifact reclaim 时保留 event log、manifest、patch 和源码 hash；
6. 离线 ETL 将数据转换为 ChatML JSONL/Parquet。

### 数据切分

必须在数据生成前冻结 train/dev/held-out，并按以下维度分组切分：

- operator 与完整 contract；
- GPU architecture；
- dtype/format；
- shape cluster；
- 原始实现/source lineage。

同一个 kernel 的不同轮次、相邻 patch 和近重复 shape 不能跨 split。Held-out 的代码、patch、case ID 和衍生 Skill 不得进入训练或 retrieval。

## 4. 指标与最终评估

主指标是：

```text
One-shot Valid Optimization Rate =
第一次输出的 patch
  可以应用
  AND compile pass
  AND correctness pass
  AND benchmark valid
  AND speedup 超过测量噪声
```

必须同时报告：

- Patch apply rate；
- Compile Pass@1；
- Correctness Pass@1；
- Valid optimization success@1；
- One-step error recovery rate；
- Balanced regression repair rate；
- independently verified geomean/median speedup；
- 每个 language-architecture、format 和 shape regime 的结果；
- language-architecture 等权 macro average 与按任务量计算的 micro average；
- generated token、GPU evaluations 和 GPU minutes per valid success。

`Pass@1` 是第一版主指标，`Pass@k` 只能作为辅助。

Base 与 SFT 必须使用完全相同的：

- held-out task；
- 初始源码和 baseline；
- Skill 与可见上下文；
- Harness；
- 解码参数和一次生成预算；
- GPU、ROCm 和 compiler；
- correctness/benchmark 重复次数。

验收建议：

```text
correctness_SFT >= correctness_Base - 2 percentage points
95% paired CI(valid_success_SFT - valid_success_Base) > 0
verified geomean speedup 不退化
GPU minutes per valid success 不显著增加
```

每个准确 GPU SKU 和 language-architecture 组合必须独立出报告，不能让某种
语言或某个平台的提升掩盖其他组合的退化。

## 5. 防止模型能力退化

1. 第一版使用 LoRA/adapter，不直接全参数训练；
2. 使用较小学习率、1–2 epoch，并根据 frozen dev early stop；held-out 仅用于最终评测；
3. 混入约 10%–20% 高质量通用 coding 数据；
4. 对源码和 patch 做文本、AST 与 diff 级去重；
5. 平衡 Triton/HIP/FlyDSL/Gluon、gfx942/gfx950 及不同 format/shape regime；
6. 保留 Base checkpoint，通过角色选择是否加载 kernel adapter；
7. 固定非 kernel coding benchmark，检测普通代码生成、bug fix、instruction following、diff 与 tool-call 格式；
8. 至少运行多个训练 seed，避免把随机波动判断为能力提升。

如果 kernel 指标提升但通用 coding benchmark 明显下降，则不能晋升该 checkpoint。

### 通用 Coding Replay 数据来源

优先使用以下公开数据：

1. [`SWE-Gym/SWE-Gym`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym)
   - 2,438 个真实 Python repository tasks；
   - 项目采用 Apache-2.0；
   - 包含可执行环境、测试和成功 Agent trajectory；
   - 其 repository patch 形式与本项目最接近，应作为主要通用 replay 来源。
2. [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
   - 约 500 万条 Coding SFT 样本；
   - 数据集采用 CC BY 4.0；
   - 包含 unit tests、`tests_execution_status` 和 `average_test_score`；
   - 只抽取测试全部通过、`average_test_score=1.0` 且长度/语言合适的样本。

对于 1,000 条 Kernel SFT 样本，加入约 125–250 条通用 Coding replay
即可形成约 10%–20% 的最终训练占比。建议其中约 70% 来自 SWE-Gym 的成功
patch trajectory，约 30% 来自 OpenCodeInstruct 的 execution-passed 样本。

通用数据必须：

- 使用独立 `task_type=general_coding_replay`，避免与 Kernel `unified diff`
  schema 混淆；
- 保留原始数据集 ID、版本、许可证、上游 repository 和测试状态；
- 检查上游 repository 的独立许可证，不假设数据集许可证覆盖所有代码；
- 与本项目 held-out benchmark 做题目、代码和 AST 级去重；
- 不把 HumanEval、MBPP、LiveCodeBench 或最终通用能力评测题混入训练；
- 在 dataset card 和模型说明中保留 Apache-2.0/CC BY 4.0 的归属信息。

## 6. 防止模型作弊和数据污染

### 执行边界

- 只允许修改 `source_file_path` 中声明的 kernel 文件；
- Harness、oracle、config、metadata 和 tests 必须只读；
- 拒绝越界路径、symlink、binary artifact 和动态修改验证逻辑；
- 如果目标是训练 kernel coding，禁止直接调用现成 AITER/Torch reference 绕过实现。

### 独立验证

- 在全新 workspace 应用 patch；
- correctness 通过后才允许 benchmark；
- 使用隐藏 shape、随机 seed、随机输入、非对齐尺寸和边界值；
- 检查输入是否被非法修改；
- 验证目标 kernel 确实 launch；
- 检测固定输出、缓存结果、预计算查表和 reference 调用；
- 最终由 `verify_engineer` 和 Director 独立重测。

### 性能真实性

- 每张 GPU 串行评测；
- 固定 clocks/runtime 环境；
- 预热并重复测量；
- 保存完整 timing receipt；
- 过滤 contention、outlier 和高方差结果；
- 只与冻结 baseline 比较；
- 对代码尺寸、编译时间和 shape specialization 设置明确约束。

### 数据完整性

- `worker_result.json` 和 Engineer 自报 speedup 只能作为辅助信息；
- 正样本必须由 independent verify 提供标签；
- held-out 数据不能进入训练、Skill、KB 或 retrieval；
- 保存 model、tokenizer、代码、ROCm、compiler、task set 和 dataset 版本；
- 所有源码和 patch 使用 hash 关联，确保 patch 与输入版本严格匹配。

## 7. `kernel_workflow` 与 `perf_knowledge` 的学习策略

这两个目录可以帮助模型，但不应直接作为 `output.patch` 的 SFT target。它们没有
形成“指定 parent source → independent verified patch”的监督对，静态文档中的
代码片段和性能数字也没有针对当前冻结 Harness 重新验证。

### 7.1 适合作为 Skill/Retrieval 上下文

优先允许：

- `kernel_workflow/knowledge/triton_optimization.md`
- `kernel_workflow/knowledge/hip_optimization.md`
- `kernel_workflow/knowledge/amd_instinct.md`
- `kernel_workflow/knowledge/optimization_strategies.md`
- `kernel_workflow/knowledge/profiling_guide.md`
- `perf_knowledge/operators/<operator>/{overview,tuning,numerics,fusion}.md`
- `perf_knowledge/operators/<operator>/backends/*.md`
- `perf_knowledge/languages/**`
- `perf_knowledge/hardware/**`
- `perf_knowledge/optimization/**`
- `perf_knowledge/quantization/**`
- `perf_knowledge/profiling/**`
- `perf_knowledge/expert_skills/skills/*/skill.md` 中当前环境验证状态为
  `validated` 的内容。

这些内容用于给模型提供：

- AMD 架构和 dtype/quantization 约束；
- Triton/HIP/FlyDSL/CK 的实现模式；
- bottleneck 到优化技术的映射；
- MFMA、LDS、vectorization、occupancy、pipeline 等编码知识；
- profile 指标的解释方法。

如果某条 Skill 被放入训练样本输入，必须记录准确路径、内容 hash、Git SHA、
frontmatter 中的 architecture/dtype/operator 限制以及检索分数。训练与评测时应
使用相同的 retrieval policy。

### 7.2 条件使用

- `kernel_workflow/knowledge/learned/*.md`：只能在排除 dev/held-out 来源后作为
  retrieval prior；其中的历史 speedup 不能作为标签。
- `perf_knowledge/expert_skills/tuning/**`：属于 hash-pinned vendored 内容，
  使用前应单独检查上游许可证和版本。
- `kernel_workflow/knowledge/wrapper_optimization.md`：只有允许修改 wrapper 时
  才能注入；第一版 kernel-file-only GEMM 可排除。
- `perf_knowledge/workflows/authoring_a_kernel_with_geak.md` 和
  `gemm_tuning_workflow.md`：可以抽取编码知识，但应去掉控制面、测试和
  backend 选择逻辑。

### 7.3 不应进入训练或 held-out retrieval

- `kernel_workflow/roles/**`：包含 Director、TechLead、Verifier、预算和停止等
  控制决策，不属于 Coding Policy；
- `kernel_workflow/kernel_lane.js`、workflow orchestrator 和
  `kernel_workflow/scripts/**`：暴露编排、测试和评测实现；
- Harness、oracle、unittest、隐藏 case、case ID 和预期输出；
- `perf_knowledge/case_studies/**`：包含具体 campaign、shape 和性能结果，容易
  造成 held-out 泄漏；
- 与 dev/held-out kernel、symbol、shape cluster、run ID 或 patch lineage
  相关的 learned card；
- 未验证、已过期或与当前 ROCm/architecture 不匹配的性能结论；
- 绝对 latency、TFLOPS 和历史 “X× faster” 数字作为 SFT label。

`kernel_workflow/scripts/kb.py` 中对 eval-dir 路径、patch 文件、Harness case ID
等内容的泄漏检查应复用于数据 ETL。知识库的原则必须是：

```text
Skill 提供候选实现知识
Harness 提供当前环境真值
independent verify 提供训练标签
静态文档和历史性能数字永远不能覆盖真实测量
```

### 7.4 许可证与版本

GEAK 仓库本身采用 Apache-2.0，但 `perf_knowledge` 包含指向或摘录自 Triton、
vLLM、CK、AITER、论文和厂商文档的第三方内容。构建 retrieval corpus 或训练
数据时必须：

- 保存 `GEAK git SHA` 和 package version；
- 保存 `perf_knowledge` corpus hash、索引生成时间和文件清单；
- 保存每个 learned/skill card 的来源、`verified_on`、`last_seen`；
- 对较长第三方代码块单独确认上游许可证；
- 在 dataset/model card 中保留 NOTICE、来源和许可证清单。

## 8. GEAK 使用期间必须记录的参数

以下字段组成可复现 SFT 样本和可信 benchmark 的最小 provenance。标记为
“隐藏评测”的字段需要保存，但不能出现在模型输入中。

### 8.1 样本、运行与切分身份

- `schema_version`、`dataset_version`、`sample_id`、`run_id`、`eval_dir`；
- `kernel_name`、`task_type`、`split`、`held_out`；
- `round`、`engineer_id`、`candidate_id`；
- operator/contract/shape/source-lineage cluster ID；
- 数据创建时间、ETL 版本和完整 manifest hash。

### 8.2 Task 与数学契约

- `mode`、`operator/op_kind`、`entry_point/target_callable`；
- `architecture/gfx`、准确 GPU SKU、language/backend；
- `language_version`、backend/runtime version、dialect 和
  `implementation_family_id`；
- 每个 case 的 `M/N/K` 或完整 dims、dtype、layout、stride、transpose；
- quantization format、packing、scale granularity、block size、output dtype；
- regime（decode/prefill/small-batch）和 workload weight 来源；
- correctness tolerance、允许的 numerical mode、input mutation policy；
- `modifiable_files`、`baseline_callable`、`baseline_frozen`；
- `COMMANDMENT.md` 内容 hash 和 oracle/Harness/reference 的 SHA256。

### 8.3 源码与 Patch lineage

- 模型实际看到的 `parent_source` 原始字节和 `source_hash`；
- 原始 kernel 路径、baseline source hash 和 canonical workspace Git commit；
- round 开始时的 workspace HEAD；
- `best_patch.diff`/`final_patch.diff` 的完整字节和 `patch_hash`；
- patch 是否可应用、应用目标版本和冲突信息；
- warm-start、历史 patch、Skill 或 AITER artifact 的来源 hash；
- 任何 wrapper、binding 或生成文件是否在允许修改范围内。

### 8.4 模型可见上下文

- `role`、`phase`、完整 prompt、system prompt 和 chat template 版本；
- model checkpoint、tokenizer revision 和 adapter 版本；
- temperature、top-p/top-k、seed、max input/output tokens、stop tokens；
- direction 的 source/id/title/prompt/specialty/focus files；
- profile、analysis、baseline、history 中实际注入模型的精确内容 hash；
- Skill/KB retrieval query、候选列表、选中条目、rank/score 和 corpus hash；
- error-recovery 中准确的失败源码、exit code、stderr 和首个失败 case；
- regression-balance 中模型可见的 per-case report 和 regression constraints。

不得把模型没有看到的 profile、direction、error 或 Skill 在 ETL 阶段补进输入，
也不得删除真实使用过的上下文后改变 `task_type`。

### 8.5 Orchestrator 与预算参数

- `budget/budget_total/budget_used`、round limit；
- `candidate_floor`、`min_improve`、`progress_delta`、`max_no_improve`；
- agent timeout、retry、tool-call limit 和最大 turn 数；
- `stopped_by`、forced replan、deadline；
- warm-start mode、DRA/research 开关、expert-skill 开关；
- GPU pool/pin 模式和每个 candidate 的 GPU assignment；
- `box_quiet`、contention 标记和 held-out KB 写入禁用状态。

这些字段通常不进入 Coding SFT prompt，但必须进入 provenance，确保不同 run 的
数据生成策略可比较。

### 8.6 GPU、软件栈与容器

- 实际物理 GPU ID、产品名、`gfx`、CU 数量和可见 GPU 列表；
- GPU pool、pin mode、lock wait time 和 idle/busy 检查结果；
- `HIP_VISIBLE_DEVICES`、`GEAK_GPU_ALLOWED`、`GEAK_GPU_REQUIRE_IDLE`；
- ROCm、driver、firmware、PyTorch、Triton、AITER、hipcc/clang、ninja 版本；
- FlyDSL、Gluon 及其依赖的准确 commit/package version；
- `PYTORCH_ROCM_ARCH`、`GPU_ARCHS`、`TORCH_EXTENSIONS_DIR`；
- container image 名称与不可变 digest；
- GEAK、任务仓库和依赖仓库 Git SHA；
- profiler 名称、版本和降级路径；
- GPU clocks、power mode、温度以及是否存在其他 GPU process。

当前 GEAK 对其中部分字段只设置但不持久化。Collector 应在每次完整 benchmark
旁写入环境快照，而不是依赖训练时重新推断。

### 8.7 Correctness 与 Benchmark 协议

- 精确 correctness、quick benchmark、full benchmark 和 profile command；
- command cwd、环境变量、return code、stdout/stderr、timeout；
- correctness seeds、随机输入分布、`random_draws`、边界 case 和 tolerance；
- 隐藏 shape/case set 版本及 hash（隐藏评测，不进入 prompt）；
- warmup 次数、测量 iterations、baseline repeats、candidate repeats；
- timer 类型、device/host timing basis、primed/unprimed 状态；
- 每次原始 latency、median/geomean/weighted 聚合方法；
- timing receipt、variance、outlier、contention 和 reliability 标记；
- graph-capture safety、memory cap、decode-no-regress 等 hard gate。

正确性随机 seed 和隐藏 case 必须记录在受保护的评测 metadata 中，但不能进入
训练 prompt、Skill、日志摘要或模型可读 workspace。

### 8.8 结果与权威标签

- `patch_applies`、`compile_pass`、`correctness_pass`、`benchmark_valid`；
- compile/correctness 的完整错误类别、exit code 和诊断文本；
- verify 的 status、每个 case 的 baseline/candidate latency 和 speedup；
- verified geomean/arithmetic/weighted speedup；
- regression constraint 是否逐项通过；
- `verify_engineer` 的独立 workspace、结果和 variance note；
- Director 的 `validation_status`、重新测量结果、arbitration note；
- `timing_basis`、timing receipt 和 kernel launch 证明；
- 是否检测到 Harness 修改、reference delegation、固定输出、缓存或预计算；
- 最终是否 committed/promoted，以及 rejection 原因。

Engineer 的 `worker_result.json`、`report.md` 和自报 speedup 只能保存在
`auxiliary_claims` 中，不能覆盖 verify/Director 标签。

### 8.9 当前优先补齐的采集缺口

1. `EVAL_DIR/sft_events.jsonl`：完整 prompt、role、phase 和 structured response；
2. `EVAL_DIR/sft_manifest.json`：run/sample/split/version 总索引；
3. `round_N/plan.json`：Engineer 执行前的 TechLead direction；
4. 内容寻址的 `parent_source`、`source_hash` 和 `patch_hash`；
5. 每个 candidate 的 verify structured return；
6. compile/correctness 的结构化 stderr、exit code 和失败 case；
7. GPU/ROCm/container/env/timing 的 benchmark 环境快照；
8. hidden correctness seed/case-set 的受保护记录；
9. model/tokenizer/adapter/chat-template/decoding 版本；
10. direction、Skill 和 retrieval corpus 的来源及 hash。

## 8.10 SFT-f 统一数据集（2026-10-06）

基于 from-scratch benchmark 结果，SFT 数据集已增强并统一格式用于 SFT-f 训练。

### 数据集

- HuggingFace: `Zhangdanyang/agent-kernel-sft-gfx942-triton-HIP`
- 本地: `/home/danyzhan/geak_sft_dataset/merged_sft_v2/train.jsonl`

### 统一格式（messages chat）

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert GPU kernel engineer..."},
    {"role": "user", "content": "<task description + kernel code>"},
    {"role": "assistant", "content": "<diff patch or complete kernel>"}
  ],
  "task_type": "...",
  "operator": "...",
  "data_source": "patch_v1|complete_impl_v1",
  "schema_version": "messages_v1"
}
```

### 数据组成

| 文件 | 样本数 | 内容 |
|------|--------|------|
| train.jsonl | 3164 | 2000 patch + 1164 complete implementation |
| dev.jsonl | 200 | Patch 优化（格式转换） |
| samples.jsonl | 2700 | 完整样本池（格式转换） |

### Task Type 分布（train.jsonl）

| 类型 | 数量 | 训练目标 |
|------|------|----------|
| direction_conditioned | 907 | 定向优化 |
| complete_implementation | 1164 | 从零写 kernel |
| cold_start | 307 | 自主优化 |
| error_recovery | 293 | 调试修复 |
| profile_guided | 293 | 性能调优 |
| regression_balance | 200 | 安全优化 |

### 新增内容

1. **完整 wave-2000 patch 数据**（2000）: 包含全部 verified patch 样本（之前仅 300 条 wave-300）
2. **complete_implementation**（1164）: 从 `phase1-production-wave-2000-v1` 的 patch 数据通过 fuzzy patch apply 得到最终 kernel，教模型从零写 kernel。从 129 扩展到 1164（利用率 58%）
3. **格式统一**: 全部从 `geak_kernel_sft_v1` 转为 `messages` chat 格式
4. **23 种 operator 覆盖**: gemm, mha, mla, paged_attention, rms_norm, rope_kv_cache, sampling, scaled_quant_gemm, fused_silu_mul 等

### 训练建议

SFT-f 应使用 `train.jsonl`（3164 samples）进行训练。`messages` 格式直接兼容 Qwen3-Coder SFT 训练框架，loss 计算在 assistant message token 上。新增的 1164 条 `complete_implementation` 样本补齐了从零实现能力，预期提升 from-scratch benchmark 通过率。patch 与 complete_impl 约 63:37 的比例平衡了优化和从零写两种能力。

## 第一版推荐执行顺序

```text
冻结任务与 split
→ 增加 append-only collector
→ 建立 language-architecture 支持矩阵
→ 阶段一采集 Triton/HIP × gfx942/gfx950 的五类样本
→ ETL、验证、去重
→ 阶段一 Qwen Coder LoRA SFT 与四组合 held-out benchmark
→ 能力退化与作弊检查
→ 阶段二加入 FlyDSL/Gluon 并扩展到八组合
→ 八组合独立指标、macro average 与阶段一回归检查
→ 决定是否进入 RL 阶段
```

## 9. Qwen3-30B-A3B LoRA SFT 实现状态

### 9.1 实现位置与执行边界

Qwen3-30B-A3B 的训练、导出和训练前后评测代码统一放在：

```text
/home/danyzhan/Lumen/experiments/GEAK-agent-coder
```

基础 checkpoint 固定为 `Qwen/Qwen3-30B-A3B`，目标机器为单节点
8×MI308X（gfx942）。代码和 Markdown 可以在数据冻结前完成，但以下操作必须等
General Coding Replay 最终数据门禁通过后才执行：

- 8-GPU SFT；
- 30B adapter merge；
- HF blockwise FP8 完整导出；
- vLLM ROCm load/generate smoke；
- GEAK Base/SFT 正式 GPU benchmark。

训练入口必须 fail closed。至少要求：

```text
accepted == 500
leakage_status == pass
package_status == ready
checksums frozen
general-coding assistant-loss-token share in [0.15, 0.20]
immutable HF revision + tokenizer SHA256 + data-manifest SHA256 pinned
```

### 9.2 LoRA 与 FP8 设计

训练采用 Transformers + Lumen FSDP2，并在一台 8-GPU 机器上使用
`EP=8`。第一版使用 `sequential` expert backend，原因是它将每个本地 expert
表示为可注入 LoRA 的普通 `nn.Linear`：

- Attention：`q_proj`、`k_proj`、`v_proj`、`o_proj`，`rank=32`，
  `alpha=64`；
- Routed expert：`gate_up_proj`、`down_proj`，`rank=8`，`alpha=16`；
- Router、embedding、norm 和 `lm_head` 冻结；
- 基础线性层通过 Lumen E4M3 `blockwise2d` 进行 FP8 计算；
- LoRA A/B 参数和 optimizer 路径保持 BF16。

等价计算为：

```text
y = FP8Linear(x, W_frozen) + BF16Linear(BF16Linear(x, A), B) * alpha/r
```

SonicMoE 当前不作为 expert-LoRA backend。它使用融合 `w1/w2` 与 grouped
kernel，未暴露 gate/up 激活边界，不能直接套用标准 LoRA 而保持数学等价。
Sonic 可以继续用于不含 expert LoRA 的对照实验，但不能标记成与 sequential
expert LoRA 相同的训练配置。

### 9.3 Checkpoint 与部署 FP8

训练 FP8 是计算精度，不等于最终部署 checkpoint：

1. 训练期间保存 BF16 LoRA adapter 和可恢复的训练 metadata；
2. 将 adapter 合并到原始 HF BF16 权重；
3. 对可量化 attention/expert 权重执行 128×128 blockwise E4M3 导出；
4. Router、embedding、norm 和 `lm_head` 保持 BF16；
5. 写入 HF safetensors index、`weight_scale_inv` 和
   `quantization_config`；
6. 通过 tensor coverage、反量化误差和 ROCm vLLM smoke 后，才生成
   deployable marker。

候选导出会对 model shards、index 和 config 计算统一 artifact SHA256。vLLM
smoke evidence 必须同时匹配该 SHA256 与 `device_arch=gfx942`，不能用其他导出
或其他机器的 smoke 结果晋升当前产物。

现有 `export_megatron_blockwise_fp8.py` 只产生 Megatron key layout，不能当作
HF/vLLM 可直接加载的部署产物。

### 9.4 GEAK 训练前后 Benchmark

主评测使用固定的 engineer-only `ToolAgentLoop`：

- Base 与 SFT 仅允许改变 `base_url`、served model 和 checkpoint identity；
- prompt、工具、turn/token/time budget、task suite、decode 参数、GPU 与
  Harness hash 必须一致；
- 性能指标使用 GEAK 的 `speedup_vs_frozen_baseline`，不能将其误写为
  `speedup_vs_torch_compile`；
- 记录逐 turn 的 token、时间、compile、correctness、独立 verify 与 speedup；
- 汇总 Pass@k、Fast@p、first-pass、turn/token cost-of-pass、
  Speedup-AUC、错误率和 paired bootstrap CI；
- General Coding Dev-80 只作为 retention gate，不能进入训练或 kernel 主分数。

### 9.5 2026-09-21 数据与实现进度

当前 Kernel Train 2,000、Kernel Dev 200、Kernel Held-out 120 和 Coding
eval-only 40 已完成对应数据路径。General Coding Replay 当前状态为：

```text
offline CPU double-replay receipts: 500
quality eligible:                  500
deduplicated pool:                 379
Dev repository overlap:           114
final accepted:                    0 / 500
package status:                    blocked
MEnvData wave-2 frozen candidates: 209
SWE-rebench C++ wave-2 candidates: 11
```

因此当前可以完成实现和 CPU 单元测试，但不得启动正式 SFT。下一数据步骤是对
wave-2 候选执行断网、禁 GPU、fresh workspace 的双回放，重新运行泄漏、精确配额、
token mix 和 checksum 审计，再冻结最终 HF 数据。

LoRA/SFT/FP8 follow-up 已完成：配置中的 base checkpoint 已修正并锁定为
`Qwen/Qwen3-30B-A3B`，训练入口默认 `allow_training=false` 且在 distributed
初始化前执行最终 replay 门禁；加入 Dev loss 的 best-adapter 选择与 early
stopping、同拓扑 adapter warm-start 身份校验、packed-expert FP8 末两维块对齐
检查，以及绑定 artifact SHA256 的 gfx942 vLLM 晋升门禁。配置化复用改造后，
容器内完整测试集 87 项通过；仍未运行任何 GPU 训练、30B 导出或 benchmark。

### 9.6 配置化复用边界

训练实现已拆分为 hardware、recipe 和 data 三类配置。保持
`Qwen/Qwen3-30B-A3B` backend 不变时，可以只通过 YAML 切换：

- MI308/MI300 `gfx942`（E4M3 FNUZ）、MI350 `gfx950`（E4M3 OCP）或显式
  ROCm BF16 profile；
- 本地 JSONL、本地 Hugging Face layout 或已缓存且 revision-pinned 的 HF
  dataset；
- LoRA rank/alpha/dropout、支持的 attention/expert target 子集、batch、
  sequence length 与 sampling policy。

启动前会检测实际 `gcnArchName`、拓扑与精度能力；不支持 FP8 时不会静默回退。
原始数据先由 `data-build` 映射为 canonical schema，执行 split/provenance/
checksum/replay policy，生成实际按 sampling plan 排列的 tokenized JSONL 和
只读 manifest。训练再校验 manifest、tokenizer 和训练文件 SHA256。GEAK 的
500 条双回放及泄漏门禁仍由 `geak_mi308_release_v1` 强制，不会影响其他普通
coding SFT 数据，也不能被普通数据 policy 绕过。

该配置化边界不包括更换模型架构。若改用其他模型，仍需对应的 Lumen model
backend、MoE expert sharding 和 LoRA target contract，不能仅靠 YAML 安全切换。
