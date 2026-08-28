# Qwen3-30B-A3B checkpoint conversion

This directory contains only the model-specific layer needed by Megatron's
checkpoint converter. Generic converter modules (`saver_base`, `schema_base`,
and `utils`) are imported from the bundled Megatron-LM checkout.

## Layout

- `convert_hf_to_megatron.sh`: converts HF safetensors to TP=1, PP=1, EP=8.
- `loader_qwen3_moe_hf.py`: streams Qwen3 tensors into converter messages.
- `saver_qwen3_moe.py`: writes those messages into MCore checkpoints.
- `schema_qwen3_moe.py`: maps Qwen3 fields to MCore module paths.
- `qwen3_moe_mapping.py`: pure QKV, SwiGLU, and expert-sharding helpers.
- `tests/`: unit tests for tensor layouts and EP partitioning.

## Usage

Run inside the benchmark container:

```bash
bash checkpoint/convert_hf_to_megatron.sh
```

Defaults:

- HF input: `/nobackup/model/Qwen3-30B-A3B`
- output: `/nobackup/checkpoints/Qwen3-30B-A3B-tp1-pp1-ep8`
- Megatron-LM: `/workspace/Megatron-LM`

Override these with `HF_DIR`, `SAVE_DIR`, and `MEGATRON_PATH`.
