# Qwen3-30B-A3B SonicMoE expert kernel 调用路径

默认 BF16 Megatron 路径：`MOE_IMPL=sonic`，`SONIC_MOE_GEMM_BACKEND=triton`，`SONIC_MOE_GROUPED_GEMM_BACKEND=triton`。

对应 AITER 树：`Lumen/third_party/aiter`（训练时 `run_docker.sh` bind-mount 该目录）。行号以当前工作区文件为准。

SonicMoE **只替换** `MoELayer.experts`。Router、permute、EP All-to-All 仍由 Megatron 完成；token 进入 expert 时已经是 expert-major。

形状（EP=8）：`E=16` local experts，`H=2048`，`I=768`（SwiGLU，`w1` 的 N=`2I=1536`），无 bias。

---

## 0. 装配

| 步骤 | 文件 | 行号 |
|------|------|------|
| `MOE_IMPL=sonic` → `--lumen-sonic-moe` | `examples/qwen3-30b-a3b/run_qwen3_30b_a3b_megatron.sh` | 21–29 |
| GEMM backend 默认 `triton` | `examples/qwen3-30b-a3b/run_docker.sh` | 129–132 |
| `LumenConfig.enable()` 打补丁 | `lumen/config.py` | 304–305，526–530 |
| `MoELayer.experts = SonicMoEExperts` | `lumen/modules/sonic_moe.py` | 577–585 |

`run_docker.sh` 相关环境变量：

- `SONIC_MOE_GEMM_BACKEND` 默认 `triton`（选 AITER SonicMoE，不是 FlyDSL）
- `SONIC_MOE_GROUPED_GEMM_BACKEND` 默认 `triton`（选 Triton grouped GEMM，不是 hipBLASLt）
- `SONIC_MOE_USE_QWEN3_TUNED_GEMM` 默认 `1`

---

## 1. 进入 expert

```
MoELayer.forward
  → dispatcher.dispatch
  → SonicMoEExperts.forward          lumen/modules/sonic_moe.py:357
      → _profiled_experts            lumen/modules/sonic_moe.py:53
          record_function "SonicMoE.experts.fwd"
          → moe_pre_routed_inputs    aiter/ops/triton/sonicmoe.py:7–11 (re-export)
                                     _triton_kernels/moe/sonicmoe/__init__.py:573
            → _UpProjection.apply    __init__.py:630
            → _DownProjection.apply  __init__.py:649
  → dispatcher.combine
```

`SonicMoEExperts.forward`（`lumen/modules/sonic_moe.py` 418–438）调用：

```python
moe_pre_routed_inputs(
    hidden,
    permuted_probs.reshape(-1).float(),
    counts,
    self.w1, None, self.w2, None,
    stream,
    SonicMoEActivationType.SWIGLU,
    False,
    True,   # concat_layout：Megatron linear_fc1 是 gate/up 拼接
)
```

权重布局：Megatron `[E, I, H]` 转成 Sonic grouped `[E, H, I]`，`grouped_weight_layout=True`。

预路由标志：`inputs_are_pre_routed=True`，`A_idx=None`，identity gather，无 bias。

Python 入口 re-export：`third_party/aiter/aiter/ops/triton/sonicmoe.py` L7–11。

---

## 2. Expert FWD（`SonicMoE.experts.fwd`）

每层 2 次 GEMM + SwiGLU + router 加权。

### 2.1 `h = x @ w1`

| 层级 | 文件 | 行号 |
|------|------|------|
| 调用 `grouped_gemm(x, w1, …)` | `_triton_kernels/moe/sonicmoe/__init__.py` `_UpProjection.forward` | 132–139 |
| backend 分流，默认 `triton` | `grouped_gemm_triton.py` `grouped_gemm()` | 534–587 |
| fwd GEMM 实现 | `grouped_gemm_triton.py` `_grouped_gemm_triton()` | 816 |
| launch | 同上 | 907–917（Qwen3 tuned 走 910） |
| **kernel** | `@triton.jit _grouped_gemm_kernel` | **195–196** |

`A_is_transposed=False`，不进 `_grouped_gemm_dw`。

### 2.2 SwiGLU：`silu(gate) * up`

| 层级 | 文件 | 行号 |
|------|------|------|
| `activation_fwd(h, I, "swiglu", concat_layout)` | `__init__.py` | 142 |
| launch | `activation_kernels.py` `activation_fwd()` | 306–327 |
| **kernel** | `@triton.jit _glu_fwd_kernel` | **10–11** |

`CONCAT_LAYOUT=True`。

### 2.3 `y = a @ w2`

| 层级 | 文件 | 行号 |
|------|------|------|
| `grouped_gemm(a, w2, …)` | `__init__.py` `_DownProjection.forward` | 295 |
| 同上进 `_grouped_gemm_triton` | `grouped_gemm_triton.py` | 816 → 910 |
| **kernel** | `_grouped_gemm_kernel` | **195–196** |

### 2.4 `o = y * router_score`

预路由 `K=1`，identity gather。

| 层级 | 文件 | 行号 |
|------|------|------|
| `_router_forward(...)` | `__init__.py` | 301–310 |
| 转调 | `forward.py` `_router_forward` | 10–30 |
| launch | `reduction_over_k_gather.py` `token_gather_and_sum_varlen_K_triton()` | 146–184 |
| **kernel** | `@triton.jit token_gather_sum_kernel` | **61–62** |

预路由路径不跑 `_topk_softmax_fwd` / routing metadata kernel。

---

## 3. Expert BWD（`SonicMoE.experts.bwd`）

Autograd 先 `_DownProjection.backward`（`__init__.py` 332），再 `_UpProjection.backward`（`__init__.py` 178）。

无 bias：**不打** `db1_kernel` / `db2_and_ds_kernel`。  
预路由：**不打** `_token_broadcast_backward`（`__init__.py` 235 直接 `dx_reduced = dx_expanded`）。

### 3.1 Down dgrad：`u = dout @ w2^T`

| 层级 | 文件 | 行号 |
|------|------|------|
| `_down_projection_backward_act(...)` | `__init__.py` | 358–374 |
| `grouped_gemm(dout, w2, B_is_transposed=True)` | `backward.py` | 256–262 |
| `_grouped_gemm_triton`（B 先 transpose） | `grouped_gemm_triton.py` | 577–587 |
| **kernel** | `_grouped_gemm_kernel` | **195–196** |

### 3.2 重算 SwiGLU + dSwiGLU

| 层级 | 文件 | 行号 |
|------|------|------|
| `activation_fwd(...)` 重算 `a` | `backward.py` | 264 |
| **kernel** | `_glu_fwd_kernel` | **10–11**（launch：`activation_kernels.py` 314） |
| `activation_bwd(...)` | `backward.py` | 269 |
| **kernel** | `@triton.jit _glu_bwd_kernel` | **80–81**（launch：`activation_kernels.py` 362） |

### 3.3 Down wgrad：`dW2 = a^T @ dy`

| 层级 | 文件 | 行号 |
|------|------|------|
| `grouped_gemm(a, dy, A_is_transposed=True)` | `__init__.py` | 385–391 |
| 因 `A_is_transposed and B.dim()==2` 走 dw | `grouped_gemm_triton.py` `_grouped_gemm_triton` | 830–833 |
| `_grouped_gemm_dw()` | 同文件 | 921 |
| launch | 同文件 | 996–1006（tuned 走 999） |
| **kernel** | `@triton.jit _grouped_gemm_dw_kernel` | **398–399** |

### 3.4 Up dgrad：`dx = dh @ w1^T`

| 层级 | 文件 | 行号 |
|------|------|------|
| `_up_projection_backward_act(...)` | `__init__.py` | 205–214 |
| `grouped_gemm(dh, w1, B_is_transposed=True)` | `backward.py` | 203–209 |
| **kernel** | `_grouped_gemm_kernel` | **195–196** |

`db1 is None`，不进 `backward.py` 211–219 的 `db1_kernel`。

### 3.5 Up wgrad：`dW1 = x^T @ dh`

| 层级 | 文件 | 行号 |
|------|------|------|
| `grouped_gemm(x, dh, A_is_transposed=True)` | `__init__.py` | 222–233 |
| `_grouped_gemm_dw()` | `grouped_gemm_triton.py` | 921 → 999 |
| **kernel** | `_grouped_gemm_dw_kernel` | **398–399** |

---

## 4. 落到 GPU 的 kernel 清单

每层 expert：**6 次 GEMM + 3 次激活/加权**。

| 步骤 | 调用点 | 实际 kernel | 文件:行 |
|------|--------|-------------|---------|
| up fwd GEMM | `__init__.py:132` | `_grouped_gemm_kernel` | `grouped_gemm_triton.py:196` |
| SwiGLU fwd | `__init__.py:142` | `_glu_fwd_kernel` | `activation_kernels.py:11` |
| down fwd GEMM | `__init__.py:295` | `_grouped_gemm_kernel` | `grouped_gemm_triton.py:196` |
| router 加权 | `__init__.py:301` → `forward.py:10` | `token_gather_sum_kernel` | `reduction_over_k_gather.py:62` |
| down dgrad | `backward.py:256` | `_grouped_gemm_kernel` | `grouped_gemm_triton.py:196` |
| 重算 SwiGLU | `backward.py:264` | `_glu_fwd_kernel` | `activation_kernels.py:11` |
| SwiGLU bwd | `backward.py:269` | `_glu_bwd_kernel` | `activation_kernels.py:81` |
| down wgrad | `__init__.py:385` | `_grouped_gemm_dw_kernel` | `grouped_gemm_triton.py:399` |
| up dgrad | `backward.py:203` | `_grouped_gemm_kernel` | `grouped_gemm_triton.py:196` |
| up wgrad | `__init__.py:222` | `_grouped_gemm_dw_kernel` | `grouped_gemm_triton.py:399` |

次数汇总：

| kernel | 次数/层 | 用途 |
|--------|---------|------|
| `_grouped_gemm_kernel` | 4 | up/down fwd，down/up dgrad |
| `_grouped_gemm_dw_kernel` | 2 | down/up wgrad |
| `_glu_fwd_kernel` | 2 | fwd 一次 + bwd 里重算一次 |
| `_glu_bwd_kernel` | 1 | SwiGLU 反向 |
| `token_gather_sum_kernel` | 1 | router score 加权 |

---

## 5. GEMM backend 分流（不是这条默认路径）

`grouped_gemm()` 在 `grouped_gemm_triton.py:564` 读 `SONIC_MOE_GROUPED_GEMM_BACKEND`（有 `A_scale`/`B_scale` 时强制 `triton`）：

| backend | 入口行 | 底层 |
|---------|--------|------|
| `triton`（默认） | 577 | `_grouped_gemm_kernel` / `_grouped_gemm_dw_kernel` |
| `multistream` | 602 | `aiter::hipb_multistream_mm` → 每个 expert 一次 `hipblasLtMatmul` |
| `hipblaslt` / `auto` | 615 | `aiter::hipb_grouped_mm`；`auto` 失败再 multistream，再 triton |

Qwen3-30B-A3B 上 **`hipb_grouped_mm` 选不出 algo**（空 expert 被写成 N=1，gfx950 拒绝）。早期训练 profile 里头号算子是 `aiter::hipb_multistream_mm`（Self CUDA 742 ms），后来默认切到 Triton。

无论 GEMM 走哪条，SwiGLU 和 `token_gather_sum_kernel` 始终是 Triton。

`SONIC_MOE_GEMM_BACKEND=flydsl` 时整条 expert MLP 不再进 AITER SonicMoE，见 `lumen/modules/sonic_moe.py:394` 和 [`flydsl-sonic-moe-runbook.md`](flydsl-sonic-moe-runbook.md)。

当前 `third_party/aiter` 的 `grouped_gemm()` **已经**接受 `A_scale`/`B_scale`（blockwise FP8）。默认 BF16 训练不走这条；官方 `blockwise2d` expert FP8 走 Lumen `grouped_fp8_expert_mlp`，不在本文 BF16 路径上。

---

## 6. 关键文件一览

相对 `Lumen/`：

| 文件 | 内容 |
|------|------|
| `lumen/modules/sonic_moe.py` | Megatron adapter，`moe_pre_routed_inputs` 入口 |
| `third_party/aiter/aiter/ops/triton/sonicmoe.py` | Python re-export |
| `third_party/aiter/aiter/ops/triton/_triton_kernels/moe/sonicmoe/__init__.py` | `_UpProjection` / `_DownProjection` / `moe_pre_routed_inputs` |
| `.../sonicmoe/grouped_gemm_triton.py` | GEMM 分流与 Triton kernel |
| `.../sonicmoe/backward.py` | dgrad custom op |
| `.../sonicmoe/activation_kernels.py` | SwiGLU fwd/bwd |
| `.../sonicmoe/forward.py` | `_router_forward` |
| `.../sonicmoe/reduction_over_k_gather.py` | `token_gather_sum_kernel` |
| `third_party/aiter/aiter/ops/gradlib.py` | `hipb_grouped_mm` / `hipb_multistream_mm` 声明 |
