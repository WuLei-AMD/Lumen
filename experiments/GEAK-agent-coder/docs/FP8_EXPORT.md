# HF Blockwise FP8 Export

## Distinct artifacts

Lumen FP8 training quantizes eligible operations during forward/backward while
retaining a master checkpoint and BF16 LoRA adapters. It is not a deployable
HF FP8 checkpoint.

Export performs:

```text
HF base + BF16 adapter
→ merge adapter deltas
→ blockwise weight quantization
→ HF safetensors + scales + quantization metadata
→ validation
→ optional deployable marker
```

## Quantization policy

- format: profile-controlled E4M3 FNUZ on gfx942 or E4M3 OCP
  (`e4m3fn`) on gfx950;
- weight block: 128×128;
- activation scheme: dynamic;
- quantized: eligible attention and routed-expert matrix weights;
- BF16: router, embeddings, norms, `lm_head`, unsupported/misaligned tensors;
- scale tensor: inverse dequantization scale associated with each weight.

The exporter writes to a candidate directory first. It must not mutate the
base checkpoint or adapter checkpoint.

## Gates

The candidate validation report covers:

- expected tensor coverage and no unknown omissions;
- block alignment and scale shapes;
- finite scales and dequantization error thresholds;
- complete config/tokenizer/generation metadata;
- safetensors index consistency;
- a SHA256 identity over model shards, index, and model config;
- adapter merge equivalence on a fixed tiny/reference fixture.

A real 30B checkpoint remains `candidate` until ROCm vLLM loads it and
generates from fixed smoke prompts. Only that external smoke can create the
deployable marker. Smoke evidence must match the manifest's `target_arch` and
`fp8_format` and carry the exact candidate artifact SHA256; evidence from a
different export or AMD architecture is rejected.

The existing Megatron `model_fp8.pt` exporter uses Megatron parameter names
and is not accepted as an HF/vLLM deployment checkpoint.
