# Qwen3-30B-A3B LoRA Design

## Scope

The model backend is `Qwen/Qwen3-30B-A3B`. Hardware/topology comes from an AMD
profile; the production release profile remains one 8×MI308X node. The
implementation adapts attention and routed experts while preserving the base
router and vocabulary behavior.

## Adapter topology

| Component | Modules | Rank | Alpha | Precision |
|---|---|---:|---:|---|
| Attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 32 | 64 | BF16 |
| Routed expert | `gate_up_proj`, `down_proj` | 8 | 16 | BF16 |
| Router | `mlp.gate` | frozen | — | BF16 |
| Embedding, norms, lm_head | — | frozen | — | BF16 |

For a frozen base linear layer, the adapter computes:

```text
y = base_fp8(x, W) + (alpha / rank) * (x @ A.T) @ B.T
```

`A` is initialized with Kaiming uniform and `B` with zeros, so injection does
not change the initial model output.

## Injection order

Qwen3 stores the original 128 expert weights in packed tensors. With `EP=8`,
`shard_moe_experts()` materializes 16 local experts on each rank. Expert LoRA
must therefore be injected after expert sharding, when the sequential backend
has created `_LocalExpertMLP.gate_up_proj` and `.down_proj` `nn.Linear`
modules, and before Lumen quantization/FSDP wrapping.

The required order is:

```text
load HF base
→ shard routed experts across EP ranks
→ freeze all base parameters
→ inject attention and local-expert LoRA
→ enable Lumen FP8 on eligible frozen base linears
→ wrap local experts and layers with dual-mesh FSDP2
```

The optimizer must receive only parameters identified by the adapter audit.
An empty adapter set, an unexpected trainable base parameter, or missing
expert targets is fatal.

Ranks, alpha, dropout and the supported attention/expert target subsets are
recipe fields. The strict GEAK release recipe pins the values above; reusable
recipes may change them while validation still rejects unknown Qwen3 module
names.

## Why sequential experts

The sequential backend represents each expert projection as ordinary
`nn.Linear`, which lets the existing Lumen linear FP8 path quantize the frozen
base while adapter matrices remain BF16.

SonicMoE uses fused `w1`/`w2` parameters and a grouped kernel. Standard LoRA
cannot be added after the fused expert output because `gate_up_proj` changes
the input to the nonlinear SiLU/gating operation. Sonic expert LoRA therefore
requires a separate mathematically aware fused implementation and is not part
of the first production path.

## Checkpoint contract

Adapter checkpoints contain:

- adapter tensors only;
- module-to-rank/alpha mapping;
- base model identity and revision;
- tokenizer/chat-template SHA256;
- frozen data-manifest SHA256;
- world and EP topology.

Loading rejects a different base identity, topology, adapter shape, or module
set. Merge/export always starts from the original HF base plus an audited
adapter checkpoint; training-time FP8 caches are not deployment artifacts.

`resume_adapter_dir` is an adapter warm start on the same topology. It restores
LoRA tensors after checking all pinned identities; it intentionally does not
claim exact optimizer/data-iterator continuation.
