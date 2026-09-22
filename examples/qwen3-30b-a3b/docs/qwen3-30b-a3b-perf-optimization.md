# Qwen3-30B-A3B MoE 训练性能优化

> **粘贴方式**：Confluence Cloud 编辑器支持直接粘贴 Markdown 并自动转换。
> 全选本文件内容 → 在空白页面里 `Ctrl+V` 即可。
> 图片需要手动补：把 `docs/charts/*.png` 拖进对应的「📊 图 N」占位行。
> 想要连图带排版一次到位，用 `publish_to_confluence.py`（走 REST API）。

---

**一句话结论**：在 8×MI350X 上，4 项优化（CK Attention、Triton Grouped GEMM、EP All-to-All Overlap、`CUDA_DEVICE_MAX_CONNECTIONS=8`）把 Qwen3-30B-A3B + SonicMoE Megatron 预训练单步从 **24.71 s 降到 17.47 s（−29.3%）**；再加上默认的 **attention-only CUDA Graph**（`CUDA_GRAPH_SCOPE=attn`）到 **12.72 s（相对基线 −48.5%）**，吞吐 **10.36 → 20.13 samples/s**，loss 对齐。

---

## 1. 背景与目标

Qwen3-30B-A3B 是 30B 总参 / 2.7B 激活的 MoE 模型（48 层，hidden 2048，128 experts，top-k 8，expert FFN 768）。训练栈为 **Megatron-LM(ROCm) + Lumen + AITER SonicMoE**。

关键结构认知：**SonicMoE 只替换 `MoELayer.experts`**，Megatron 仍负责 router、permute 和 3 次 EP All-to-All。因此优化必须把「专家 GEMM」与「通信/调度」分开处理——这也是后续 4 项优化各自的着力点。

> ℹ️ 本文所有数据均为**本机真实权重复现**（加载 HF 转换的 tp1-pp1-ep8 checkpoint + FineWeb 真实语料），非 mock / 随机初始化。早期 mock 数据跑出的 30.8 s 基线不具可比性，已从本文剔除。

## 2. 测试环境

| 项 | 值 |
|---|---|
| 硬件 | 1 节点 8× AMD Instinct MI350X (gfx950) |
| 镜像 | `zhangdanyangamd/lumen:qwen3-30b-a3b-350x-pretrain260829-multistream` |
| 软件栈 | ROCm 7.2 / PyTorch 2.9.1 / Megatron-LM(ROCm) / TransformerEngine(ROCm, CK) / AITER |
| 并行策略 | TP=1, PP=1, CP=1, **EP=8**, ETP=1, DP=8 |
| 精度 | BF16 |
| 序列长度 | 4096 |
| MoE 实现 | `MOE_IMPL=sonic`（`--lumen-sonic-moe`），dispatcher = `alltoall` |
| 权重 | `/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8` |
| 数据 | FineWeb sample 10BT（26624 docs） |
| 基础开关 | `--overlap-grad-reduce --overlap-param-gather --use-distributed-optimizer --moe-permute-fusion` |

**两套测量口径**（本文所有对比都严格区分，避免混用）：

| 口径 | 配置 | 统计窗口 | 用途 |
|---|---|---|---|
| **A. 小 batch 快速 A/B** | MBS=1, GBS=16（2 microbatch），10 step | step 4–10 中位数 | 逐项优化归因，单轮约 4.5 分钟 |
| **B. 大 batch 端到端** | MBS=2, GBS=256（16 microbatch），20 step | step 11–20 中位数 | 对齐生产配置，报告最终收益 |

> 💡 统一使用**中位数**而非均值：MI350X 上单步存在 ±10% 抖动，均值容易被个别慢步带偏。同时前 3 步（warmup / autotune / checkpoint 加载）一律丢弃。

---

## 3. 总体收益（大 batch，生产配置）

MBS=2, GBS=256, seq=4096, 20 step, steps 11–20 中位数。

| 指标 | 优化前<br/>(Triton Attn + multistream GEMM) | 4 项全开<br/>(无 graph) | **4 项 + attn CUDA Graph**<br/>(当前默认) |
|---|---|---|---|
| **单步时间（中位）** | 24.71 s | 17.47 s | **12.72 s** |
| **吞吐** | 10.36 samples/s | 14.66 samples/s | **20.13 samples/s** |
| 算力利用（中位） | 122.0 TFLOP/s/GPU | 172.7 TFLOP/s/GPU | **237.2 TFLOP/s/GPU** |
| 显存占比 (mem usages) | 0.609 | 0.710 | 0.692 |
| step 20 lm loss | 2.3847 | 2.3847 | 2.3846 |

📊 **图 1** — `01-e2e-bigbatch.png`：大 batch 端到端收益（单步时间 / 吞吐 / 算力利用率）

**日志**：

```text
优化前: examples/qwen3-30b-a3b/results/qwen3-30b-a3b-sonic-real-repro-multistream-priority0-seq4096-mbs2-gbs256.log
优化后: examples/qwen3-30b-a3b/results/qwen3-30b-a3b-sonic-real-repro-mbs2-gbs256-ck-triton-overlap-conn8-seq4096-mbs2-gbs256.log
```

---

## 4. 逐项优化归因（小 batch A/B）

MBS=1, GBS=16, seq=4096, 10 step, steps 4–10 中位数。每一行都是在前一行基础上**累加一项**开关，其余保持不变。

| 阶段 | 累加的优化 | 单步中位 (s) | 吞吐 (samples/s) | TFLOP/s/GPU | 相对上一步 | 相对基线 |
|---|---|---|---|---|---|---|
| **A0** | 基线：Lumen Triton Attention + hipBLASLt multistream Grouped GEMM | 2.27 | 7.04 | 82.9 | — | — |
| **A1** | + Attention → AITER CK/`csrc` (`fmha_v3`) | 1.94 | 8.24 | 97.0 | **−14.5%** | −14.5% |
| **A2** | + Grouped GEMM → Triton（Qwen3 调优配置） | 1.72 | 9.28 | 109.3 | **−11.3%** | −24.2% |
| **A3** | + `--overlap-moe-expert-parallel-comm` | 1.62 | 9.90 | 116.6 | **−5.8%** | −28.6% |
| **A4** | + `CUDA_DEVICE_MAX_CONNECTIONS=8` | **1.57** | **10.19** | **120.1** | **−3.1%** | **−30.8%** |

合计：单步 **2.27 s → 1.57 s（−30.8%，加速 1.45×）**，吞吐 **+44.7%**。全程 lm loss 稳定在 2.448，未出现数值偏移。

📊 **图 2** — `02-cumulative-steptime.png`：逐项优化的单步时间下降（标注每步降幅）

📊 **图 3** — `03-throughput-tflops.png`：逐项优化的吞吐与算力利用率爬升

📊 **图 4** — `04-waterfall.png`：瀑布图，700 ms 的节省来自哪里

---

## 5. 各优化项详解

### 5.1 Attention：Triton → AITER CK/csrc (fmha_v3)

✅ **已合入 `ba4d359`** · 收益 **−14.5%**

| | |
|---|---|
| **问题** | 默认走 Lumen 自研 Triton FlashAttention。Profile 显示 attention 前后向合计占 self CUDA **347 ms / 2.47 s ≈ 14%**，其中 backward 独占 281 ms。 |
| **改动** | `LUMEN_ATTN_BACKEND` 默认值 `triton` → `csrc`，转而调用 AITER 的 CK 实现 `aiter::fmha_v3_fwd/bwd`。 |
| **为什么有效** | CK 针对 gfx950 做了 MFMA tile 与 pipeline 调优，而 Lumen Triton kernel 主要面向通用形状。GQA（Hq=32 / Hkv=4）在 CK 上有专门的 kernel 变体。 |
| **风险** | 低。可用 `LUMEN_ATTN_BACKEND=triton` 一键回退。 |

**Kernel 级微基准**（真实形状 Q`[1,4096,32,128]` / KV`[1,4096,4,128]`，BF16 causal）：

| 算子 | Triton | AITER CK | 加速比 |
|---|---|---|---|
| Forward | 1.03 ms | **0.17 ms** | **6.1×** |
| Backward | 3.21 ms | **0.68 ms** | **4.7×** |

📊 **图 5** — `05-microbench.png`：Attention 后端 与 Grouped GEMM 后端的 kernel 微基准

### 5.2 Sonic Grouped GEMM：hipBLASLt multistream → Triton

✅ **已合入 `b4371da`** · 收益 **−11.3%**

| | |
|---|---|
| **问题** | `aiter::hipb_multistream_mm` 是 profile 里的**头号算子**：self CUDA **742 ms，占 30%**。它为每个 expert 单独 launch 一个 hipBLASLt GEMM，小 M 下 launch 开销与尾部效应严重。 |
| **改动** | `SONIC_MOE_GROUPED_GEMM_BACKEND=triton` + `SONIC_MOE_USE_QWEN3_TUNED_GEMM=1`（启用针对 Qwen3 形状预调的 BLOCK_M/N/K 配置）。 |
| **为什么有效** | Triton grouped GEMM 用**单个 kernel** 处理全部 expert，通过 device 端 `cu_seqlens` 做块到 expert 的映射，消除了 per-expert launch。 |
| **附带发现** | 原生 `hipb_grouped_mm`（非 multistream）在本模型上**直接失败**：前向 `setup_n` 会把 expert 0 伪造成 N=32768、其余为 N=1，gfx950 的 grouped tile 拒绝 N=1，报 `no hipBLASLt grouped algorithm passed isAlgoSupported`。 |

**单层 6 个 GEMM 的微基准**（E=16, TK=32768, w1 `(16,2048,1536)`, w2 `(16,768,2048)`）：

| 后端 | 单层合计 | 结论 |
|---|---|---|
| **triton (Qwen3 tuned)** | **1.78 ms** | ✅ 采用 |
| triton (autotune) | 1.79 ms | 与调优配置基本持平 |
| multistream (原方案) | 2.30 ms | 慢 29% |
| hipblaslt grouped | — | ❌ 不可用，空 expert N=1 |

### 5.3 EP All-to-All Overlap（Megatron combined 1F1B）

✅ **已合入 `eb23a8e`** · 收益 **−5.8%**

| | |
|---|---|
| **问题** | 计算侧优化到位后，**通信成为最大头**：RCCL 合计 557 ms（34%），其中 EP `all_to_all` 396 ms / 576 次。576 = 48 层 × 2 microbatch × (fwd+bwd) × 3 次（counts + dispatch + combine）。 |
| **改动** | 训练脚本增加 `--overlap-moe-expert-parallel-comm`（由 `OVERLAP_MOE_EP_COMM` 控制，默认开）。 |
| **为什么有效** | Megatron 切到 **combined 1F1B** 调度：把 microbatch *i* 的 backward 与 microbatch *i+1* 的 forward 合并执行，用另一条 microbatch 的 attention/MLP 计算来掩盖本条的 EP A2A。 |
| **代码依赖** | Megatron 要求 `forward_step(..., return_schedule_plan=True)` 返回 `GPTModel.build_schedule_plan(...)`。Lumen 的 `make_forward_step` 原先没有该参数，已补齐（`lumen/models/megatron.py`）。 |
| **代价** | 显存 +6~17%（两条 microbatch 同时驻留）。 |

> ⚠️ **该优化对 microbatch 数量高度敏感。** MBS=1/GBS=16 只有 2 个 microbatch，仅中间一段能重叠，首个 forward 与最后一个 backward 的 A2A 仍完全暴露，故只有 −5.8%；MBS=2/GBS=256 有 16 个 microbatch，暴露比例大幅下降，是大 batch 收益更高的主因。
> **切勿设置成单 microbatch**（如 MBS=2/GBS=16），此时 overlap 完全失效。

同层同 microbatch 内，token A2A 与专家 GEMM 存在数据依赖，**无法**重叠——这是选择跨 microbatch 方案的根本原因。

### 5.4 CUDA_DEVICE_MAX_CONNECTIONS：1 → 8

✅ **已合入 `843dcd5`** · 收益 **−3.1%**（改一行）

该变量控制**驱动向 GPU 同时挂载的提交队列（connection）数量**，不是 batch，也不是 NCCL 连接数。

| 取值 | 行为 | 适用场景 |
|---|---|---|
| `=1` | 所有 stream 挤在同一队列，kernel 按 launch 顺序 FIFO 执行 | **TP/SP**：强制「先发的 NCCL 先启动」，GEMM 才能叠在通信后面 |
| `>1` | 不同 stream 走不同队列，计算与通信可真正并行 | **EP A2A Overlap**：compute stream 与 comm stream 需并发提交 |

镜像 `Dockerfile` 硬编码了 `CUDA_DEVICE_MAX_CONNECTIONS=1`（TP 配方），与本任务的 EP overlap **诉求相反**。本项目 TP=1，不存在 FIFO 约束，应使用大值。

> ℹ️ **为什么选 8 而不是 Megatron 文档常写的 32？**
> Megatron README 只要求 `CUDA_DEVICE_MAX_CONNECTIONS > 1`；32 是社区在「EP overlap + CUDA Graph」复杂场景下的经验值，并非实测最优。8 恰是 CUDA 默认值，且镜像已设 `GPU_MAX_HW_QUEUES=8`——**硬件队列上限就是 8**，把 connections 开到 32 也无法获得更多真实并行。实测 8 已拿到全部收益。

### 5.5 Attention-only CUDA Graph：`CUDA_GRAPH_SCOPE=attn`

✅ **已合入默认** · 相对 4 项无 graph **−27.2%**（17.47 s → 12.72 s）

| | |
|---|---|
| **问题** | 4 项之后 CPU launch / attention 仍占墙钟。 |
| **改动** | Megatron TE `make_graphed_callables`，只 graph `_forward_attention`（`CUDA_GRAPH_SCOPE=attn`）。MoE / EP A2A 不进 graph。 |
| **为什么有效** | Attention 前后向变成 replay，砍掉大量 host launch；与 EP overlap 兼容。 |
| **风险** | 不要把 scope 扩到 expert。FlyDSL `forward_routes_training` 在 capturing stream 上会直接报错。`CUDA_GRAPH_SCOPE=none` 可回退。 |

大 batch（MBS=2, GBS=256, steps 11–20 中位，与 README 生产表一致）：

| | 4 项无 graph | + attn graph |
|---|---|---|
| 单步 | 17.47 s | **12.72 s** |
| 吞吐 | 14.66 samples/s | **20.13 samples/s** |
| TFLOP/s/GPU | 172.7 | **237.2** |
| mem usages | 0.710 | 0.692 |
| lm loss | 2.3847 | 2.3846 |

---

## 6. Profile 对比（算子级证据）

PyTorch Profiler，rank0，单个完整 train step。**注意：此 profile 对应阶段 A0 → A2**（CK Attention + Triton GEMM，尚未开启 EP overlap 与 CONN=8）。

Self CUDA 总时间：**2.473 s → 1.660 s（−32.9%）**；Self CPU：2.198 s → 1.506 s。

| 算子分组 | A0 self CUDA | A2 self CUDA | 变化 | 说明 |
|---|---|---|---|---|
| **Grouped GEMM** | 742.0 ms | **197.0 ms** | **−73%** | `hipb_multistream_mm` → `_grouped_gemm_kernel` 92.2 + `_grouped_gemm_dw_kernel` 104.8 |
| **Attention (fwd+bwd)** | 347.3 ms | **75.5 ms** | **−78%** | Triton 65.9/281.4 → `fmha_v3` 13.8/61.8 |
| RCCL 通信 `record_param_comms` | 575.0 ms | 557.2 ms | −3% | 未被本阶段优化触及，成为新瓶颈（占 34%） |
| ↳ 其中 `nccl:all_to_all` | 420.7 ms | 395.6 ms | −6% | 576 次，约 0.69 ms/次（子项，勿与上行相加） |
| elementwise / `aten::add_` | 107.7 ms | 131.5 ms | +22% | 占比被动上升，1642 次调用 |
| 其它 | ≈701 ms | ≈699 ms | 持平 | permute / sort / RMSNorm / optimizer 等 |

📊 **图 6** — `06-profile-ops.png`：关键算子 self CUDA 时间 before / after

### CPU 侧观察：hipMemcpyWithStream 占 51% 是什么

A2 profile 的 Self CPU 榜首是 `hipMemcpyWithStream`（**769 ms，51%，250 次**），其次 `hipEventSynchronize`（173 ms，96 次 = 每层一次）。

来源是 Megatron All-to-All dispatcher 的**路由元数据 D2H**，不是 token 本体。`MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize` 在侧流上异步拷贝 `tokens_per_expert`、`input_splits`、`output_splits`、`output_splits_tp`、`num_out_tokens`，随后在 `cuda_sync_point` 上 `d2h_event.synchronize()`。

目的是让下游 GroupedGEMM 的 `.tolist()` 变成纯 host 读。**已验证：单独去掉 counts 拷贝无收益**（见 7.2），真正的阻塞点是 A2A 前对 splits 的同步。

---

## 7. 已验证无效 / 放弃的方案

> 记录负结果同样重要，避免后续重复投入。

### 7.1 `--delay-wgrad-compute`

❌ **不可用**

Megatron 官方将其与 EP overlap 配套推荐（把 expert wgrad 推迟到下一层 forward 之后，扩大通信掩盖窗口）。但本栈启动即 assert 失败：

```text
AssertionError: Delaying wgrad compute is only supported with transformer_engine implementation
```

该特性依赖 TE Linear 的 `backward_dw()`；本栈使用 `--transformer-impl local`（Lumen CK Attention + SonicMoE），不具备 deferred wgrad 能力。**无性能数据**。注意与 Lumen 自己的 `--lumen-delay-wgrad` 区分，后者只服务 FP8/量化 Linear，BF16 路径用不上。

### 7.2 保持 expert counts 在 GPU（消除 D2H）

❌ **无收益，已回滚**

改动：monkey-patch `MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize`，不再拷贝 `tokens_per_expert`（NCCL 仍需的 splits 保留 D2H）；同时让 `SonicMoEExperts` 优先取 GPU counts。功能正确，10 step 跑通，loss 一致。

| 指标 | A4 基线 | GPU counts |
|---|---|---|
| 单步中位 | 1.569 s | 1.567 s |
| 吞吐 | 10.19 samples/s | 10.21 samples/s |

**原因**：`tokens_per_expert` 只有 16 个 int，拷贝本身极廉价；真正卡住 host 的是 A2A 前针对 **splits** 的 `hipEventSynchronize`，该同步依然存在。要继续优化必须让 A2A 直接接受 GPU 上的 split sizes，成本远高。

---

## 8. 后续优化候选（待验证）

按性价比排序。**实测列留空，验证后回填本表。**

| # | 优化项 | 依据 / 预期收益 | 改动量 | 风险 | 实测单步 | 状态 |
|---|---|---|---|---|---|---|
| F1 | 提高 microbatch 数（保持 MBS=1，GBS 32/64） | EP overlap 两端气泡随 microbatch 数下降；大 batch 已验证此规律 | 仅改参数 | 低（显存） | *待填* | 🟡 待验证 |
| F2 | 开启 gradient-accumulation fusion | 脚本当前写死 `--no-gradient-accumulation-fusion`；`aten::add_` 131 ms / 1642 次 | 一行 | 低，失败即回退 | *待填* | 🟡 待验证 |
| F3 | RCCL 调参（`NCCL_MIN/MAX_NCHANNELS`、`NCCL_PROTO`、`RCCL_MSCCL_ENABLE`） | EP A2A 仍 396 ms / 576 次，是当前最大单项 | 仅环境变量 | 中，可能反向 | *待填* | 🟡 待验证 |
| F4 | Attention-only CUDA Graph（`CUDA_GRAPH_SCOPE=attn`） | 削减 CPU launch；不 graph MoE/A2A | 已合入默认 | 低 | **12.72 s** | ✅ 已合入 |
| F5 | 消除 splits D2H + `hipEventSynchronize` | CPU 侧 769 ms memcpy + 173 ms sync 的真正来源（见 7.2） | 大，需改 Megatron 上游 | 高 | *待填* | ⚪ 未开始 |
| F6 | 切换 TE spec 以启用 `--delay-wgrad-compute` | 解锁 7.1；需 Sonic experts 实现 `backward_dw` | 大 | 高，可能与 CK Attention 替换冲突 | *待填* | ⚪ 未开始 |
| F7 | 修复 hipBLASLt grouped GEMM 的空 expert（N=1）问题 | 当前 GEMM 已降到 197 ms，上限有限 | 中 | 中 | *待填* | ⚪ 优先级低 |
| F8 | 空 expert skip | 收益小，且可能打乱 grouped GEMM 对齐 | 中 | 中 | *待填* | ⚪ 优先级低 |
| F9 | FP8 / MXFP8 专家 GEMM | 收益潜力最大，但需精度验证 | 大 | 高（精度） | *待填* | ⚪ 未排期 |

📊 **图 7** — `07-remaining-hotspots.png`：当前剩余热点，及对应的后续优化项编号

### 8.1 结果回填模板

复制下面这段，为每个新验证的优化项建一节：

```markdown
#### Fx. <优化项名称>   🟡 待验证

| | |
|---|---|
| 问题 | <profile 证据 / 占比> |
| 改动 | <开关、文件、代码> |
| 为什么有效 | <机理> |
| 风险 / 回退 | <> |

| 指标 | 改动前 | 改动后 | 优化比例 |
|---|---|---|---|
| 单步中位 (s) | | | |
| 吞吐 (samples/s) | | | |
| TFLOP/s/GPU | | | |
| 显存 mem usages | | | |
| 末步 lm loss | | | 需一致 |

日志：`<path>` | Commit：`<sha>`
```

---

## 9. 复现步骤

### 9.1 大 batch 端到端（生产配置）

```bash
cd /home/leiwu/Lumen

HOST_ASSET_ROOT=/dev/shm/qwen3-30b-a3b \
MODEL_PATH=/nobackup/model/Qwen3-30B-A3B \
DATA_PATH=/nobackup/data/fineweb-sample-10BT-26624.jsonl \
MEGATRON_LOAD_PATH=/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8 \
MOE_IMPL=sonic \
SONIC_MOE_GEMM_BACKEND=triton \
SONIC_MOE_GROUPED_GEMM_BACKEND=triton \
SONIC_MOE_USE_QWEN3_TUNED_GEMM=1 \
LUMEN_ATTN_BACKEND=csrc \
OVERLAP_MOE_EP_COMM=1 \
CUDA_DEVICE_MAX_CONNECTIONS=8 \
RUN_SUFFIX=mbs2-gbs256-ck-triton-overlap-conn8 \
TRAIN_STEPS=20 SEQ_LEN=4096 MBS=2 GBS=256 \
MASTER_PORT=29921 \
COMMAND='export TOKENIZER_PATH=/nobackup/model/Qwen3-30B-A3B
export LUMEN_ATTN_BACKEND=csrc
export SONIC_MOE_GROUPED_GEMM_BACKEND=triton
export SONIC_MOE_USE_QWEN3_TUNED_GEMM=1
export OVERLAP_MOE_EP_COMM=1
export CUDA_DEVICE_MAX_CONNECTIONS=8
bash run_qwen3_30b_a3b_megatron.sh' \
bash examples/qwen3-30b-a3b/run_docker.sh
```

> ⚠️ `run_docker.sh` **不会**自动把宿主环境变量透传进容器内的 Python 进程。凡是训练进程需要读取的变量（`LUMEN_ATTN_BACKEND`、`SONIC_MOE_*`、`QWEN_E2E_PROFILE_*` 等），必须在 `COMMAND` 里再 `export` 一次，如上例。

### 9.2 抓取 profile

在 `COMMAND` 中追加：

```bash
export QWEN_E2E_PROFILE_STEP=6         # 从第几步开始
export QWEN_E2E_PROFILE_STEPS=1        # 抓几步
export QWEN_E2E_PROFILE_DIR=/workspace/Lumen/examples/qwen3-30b-a3b/results/profile-xxx
export QWEN_E2E_PROFILE_TRACE=1        # 同时导出 chrome trace
```

产物：`rank0-operators.txt`（算子表）、`rank0-trace.json`（时间线）。Hook 实现见 `examples/qwen3-30b-a3b/pretrain_qwen3_30b_a3b_megatron.py` 的 `_install_e2e_profiler()`。

### 9.3 统计中位数

```python
import re, statistics
from pathlib import Path

PAT = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
    r".*?global batch size:\s*(\d+).*?lm loss:\s*([0-9.Ee+-]+)"
)

def median_step(path, lo, hi):
    txt = Path(path).read_text(errors="replace")
    ts = [float(ms) for step, _, ms, _, _ in PAT.findall(txt) if lo <= int(step) <= hi]
    gbs = int(re.search(r"global batch size:\s*(\d+)", txt).group(1))
    med = statistics.median(ts)
    return med / 1000, gbs / (med / 1000)   # (中位秒, samples/s)

# 大 batch 用 (11, 20)；小 batch 10-step 用 (4, 10)
print(median_step("examples/qwen3-30b-a3b/results/<log>.log", 11, 20))
```

### 9.4 重新生成图表

```bash
python3 -m venv /tmp/chartenv && /tmp/chartenv/bin/pip install matplotlib
/tmp/chartenv/bin/python examples/qwen3-30b-a3b/docs/make_charts.py
```

---

## 10. 变更清单

| Commit | 标题 | 涉及文件 | 开关 / 默认值 |
|---|---|---|---|
| `ba4d359` | feat(qwen3): default attention to AITER CK/csrc flash | `config_MI350X_1x8x1.sh`, `run_docker.sh` | `LUMEN_ATTN_BACKEND=csrc` |
| `b4371da` | feat(qwen3): default Sonic grouped GEMM to Triton | `run_docker.sh` | `SONIC_MOE_GROUPED_GEMM_BACKEND=triton`<br/>`SONIC_MOE_USE_QWEN3_TUNED_GEMM=1` |
| `eb23a8e` | feat(qwen3): overlap MoE EP all-to-all across microbatches | `lumen/models/megatron.py`, `run_docker.sh`, `run_qwen3_30b_a3b_megatron.sh` | `OVERLAP_MOE_EP_COMM=1` |
| `843dcd5` | feat(qwen3): default CUDA_DEVICE_MAX_CONNECTIONS to 8 | `run_docker.sh` | `CUDA_DEVICE_MAX_CONNECTIONS=8` |

分支：`dev/moe`。四项优化**均已设为默认值**，无需额外参数即可获得全部收益；每一项都保留了环境变量回退通道。

---

## 11. 附录

### 11.1 完整原始数据（小 batch，steps 4–10）

| 阶段 | 中位 (s) | 最小 (s) | 最大 (s) | samples/s | mem usages | 末步 lm loss |
|---|---|---|---|---|---|---|
| A0 基线 | 2.27 | 2.22 | 2.37 | 7.04 | 0.4798 | 2.4486 |
| A1 +CK Attn | 1.94 | 1.92 | 2.14 | 8.24 | 0.4789 | 2.4479 |
| A2 +Triton GEMM | 1.72 | 1.66 | 1.88 | 9.28 | 0.4797 | 2.4482 |
| A3 +EP Overlap | 1.62 | 1.54 | 1.64 | 9.90 | 0.5316 | 2.4479 |
| A4 +CONN=8 | 1.57 | 1.52 | 1.70 | 10.19 | 0.5323 | 2.4477 |
| *(已回滚) GPU counts* | *1.57* | *1.54* | *1.65* | *10.21* | *0.5319* | *2.4478* |

### 11.2 完整原始数据（大 batch，steps 11–20）

| 方案 | 中位 (s) | 最小 (s) | 最大 (s) | samples/s | mem usages | 末步 lm loss |
|---|---|---|---|---|---|---|
| 优化前 | 24.71 | 24.17 | 25.38 | 10.36 | 0.6092 | 2.3847 |
| 优化后（4 项，无 graph） | 17.47 | 16.11 | 19.83 | 14.66 | 0.7104 | 2.3847 |
| 4 项 + attn CUDA Graph（当前默认） | 12.72 | — | — | 20.13 | 0.692 | 2.3846 |

### 11.3 关键环境变量速查

| 变量 | 默认值 | 作用 |
|---|---|---|
| `LUMEN_ATTN_BACKEND` | `csrc` | Attention 后端；`triton` 可回退 |
| `SONIC_MOE_GROUPED_GEMM_BACKEND` | `triton` | 可选 `triton` / `multistream` / `hipblaslt` / `auto` |
| `SONIC_MOE_USE_QWEN3_TUNED_GEMM` | `1` | 启用 Qwen3 形状预调 kernel 配置 |
| `OVERLAP_MOE_EP_COMM` | `1` | 追加 `--overlap-moe-expert-parallel-comm` |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `8` | 驱动提交队列数；TP>1 时须改回 `1` |
| `CUDA_GRAPH_SCOPE` | `attn` | `none` 关闭 graph；不要 graph MoE/expert |
| `MEGATRON_OVERLAP` | `1` | DP `--overlap-grad-reduce --overlap-param-gather` |
| `SONIC_MOE_LOG_BACKEND` | `0` | 打印实际生效的 grouped GEMM 后端，用于确认开关生效 |

### 11.4 已知注意事项

- **TP>1 时必须把 `CUDA_DEVICE_MAX_CONNECTIONS` 改回 1**，否则 TP 的异步通信重叠会被破坏（Megatron 在 TP/CP>1 且非 EP-overlap 场景下会直接 assert）。
- **EP overlap 与 CUDA Graph 覆盖 MoE/MLP 互斥**，但 **`cuda_graph_scope=attn` 可以与 overlap 同开**（当前默认）。也不能与 `moe_shared_expert_overlap` 同用（Qwen3-30B-A3B 无 shared expert）。
- **不要让 microbatch 数退化为 1**，否则 EP overlap 静默失效。
- 权重与数据位于 `/dev/shm/qwen3-30b-a3b`（tmpfs），**宿主重启即丢失**，需重新下载与转换。
- HF → Megatron 转换时 TE 会把 input RMSNorm 融进 `linear_qkv`，schema 需映射为 `self_attention.linear_qkv.layer_norm_weight`，否则 checkpoint 转不出来。
