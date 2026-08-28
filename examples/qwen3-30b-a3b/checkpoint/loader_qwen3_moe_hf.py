"""Stream a Qwen3-30B-A3B Hugging Face checkpoint to Megatron's converter."""

from __future__ import annotations

import json
import os
import sys
import types

import torch
from safetensors import safe_open

from qwen3_moe_mapping import expert_range_for_ep_rank, pack_grouped_query_qkv


EXPECTED = {
    "num_hidden_layers": 48,
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
    "vocab_size": 151936,
}


def add_arguments(parser):
    group = parser.add_argument_group(title="Qwen3-MoE Hugging Face loader")
    group.add_argument("--megatron-path", type=str, default=None)
    group.add_argument("--tokenizer-model", type=str, default=None)


class SafeTensorReader:
    """Read individual tensors from a sharded HF safetensors directory."""

    def __init__(self, directory: str):
        index_path = os.path.join(directory, "model.safetensors.index.json")
        single_path = os.path.join(directory, "model.safetensors")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as stream:
                self.weight_map = json.load(stream)["weight_map"]
        elif os.path.isfile(single_path):
            with safe_open(single_path, framework="pt", device="cpu") as stream:
                self.weight_map = {name: "model.safetensors" for name in stream.keys()}
        else:
            raise FileNotFoundError(
                f"{directory} has no model.safetensors or model.safetensors.index.json"
            )
        self.directory = directory

    def tensor(self, name: str) -> torch.Tensor:
        try:
            filename = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"HF checkpoint is missing required tensor {name!r}") from error
        with safe_open(
            os.path.join(self.directory, filename), framework="pt", device="cpu"
        ) as stream:
            return stream.get_tensor(name)


def _read_config(directory: str) -> dict:
    with open(os.path.join(directory, "config.json"), encoding="utf-8") as stream:
        config = json.load(stream)
    for name, expected in EXPECTED.items():
        actual = config.get(name)
        if actual != expected:
            raise ValueError(f"config {name}={actual!r}; expected {expected!r}")
    if config.get("model_type") != "qwen3_moe":
        raise ValueError(f"expected model_type='qwen3_moe', got {config.get('model_type')!r}")
    return config


def _build_megatron_args(args, config: dict):
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    try:
        from megatron.training.arguments import parse_args, validate_args
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Unable to import Megatron; pass --megatron-path or add it to PYTHONPATH"
        )

    tokenizer_model = args.tokenizer_model or args.load_dir
    argv = [
        "qwen3-converter",
        "--use-mcore-models",
        "--num-layers",
        str(config["num_hidden_layers"]),
        "--hidden-size",
        str(config["hidden_size"]),
        "--ffn-hidden-size",
        str(config["intermediate_size"]),
        "--num-attention-heads",
        str(config["num_attention_heads"]),
        "--group-query-attention",
        "--num-query-groups",
        str(config["num_key_value_heads"]),
        "--kv-channels",
        str(config["head_dim"]),
        "--seq-length",
        "4096",
        "--max-position-embeddings",
        "4096",
        "--position-embedding-type",
        "rope",
        "--rotary-base",
        str(int(config["rope_theta"])),
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        str(config["rms_norm_eps"]),
        "--qk-layernorm",
        "--swiglu",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--num-experts",
        str(config["num_experts"]),
        "--moe-ffn-hidden-size",
        str(config["moe_intermediate_size"]),
        "--moe-router-topk",
        str(config["num_experts_per_tok"]),
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--moe-aux-loss-coeff",
        str(config.get("router_aux_loss_coef", 0.001)),
        "--moe-token-dispatcher-type",
        "alltoall",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--make-vocab-size-divisible-by",
        "1187",
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        tokenizer_model,
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "8",
        "--bf16",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-dropout-fusion",
        "--no-gradient-accumulation-fusion",
        "--use-cpu-initialization",
        "--no-initialization",
        "--mock-data",
        "--no-one-logger",
        "--transformer-impl",
        "transformer_engine",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        margs = parse_args()
        margs.world_size = 1
        margs = validate_args(margs)
    finally:
        sys.argv = old_argv
    return margs


def _metadata(args, margs, config: dict):
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = config["num_hidden_layers"]
    md.hidden_size = config["hidden_size"]
    md.seq_length = 4096
    md.num_attention_heads = config["num_attention_heads"]
    md.max_position_embeddings = 4096
    md.tokenizer_type = "HuggingFaceTokenizer"
    md.iteration = 1
    md.params_dtype = torch.bfloat16
    md.bert_binary_head = False
    md.output_layer = True
    md.position_embedding_type = "rope"
    md.linear_bias = False
    md.qkv_bias = False
    md.norm_has_bias = False
    md.swiglu = True
    md.previous_tensor_parallel_size = 1
    md.previous_pipeline_parallel_size = 1
    md.true_vocab_size = config["vocab_size"]
    md.make_vocab_size_divisible_by = 1187
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.num_experts = config["num_experts"]
    md.num_query_groups = config["num_key_value_heads"]
    md.kv_channels = config["head_dim"]
    return md


def _queue_put(queue, name: str, message: dict) -> None:
    print(f"sending {name}")
    message["name"] = name
    queue.put(message)


def _load_checkpoint(queue, args):
    if args.model_type != "GPT":
        raise ValueError("Qwen3-MoE is a GPT model")
    config = _read_config(args.load_dir)
    reader = SafeTensorReader(args.load_dir)
    margs = _build_megatron_args(args, config)
    queue.put(_metadata(args, margs, config))

    _queue_put(
        queue,
        "embeddings",
        {"word embeddings": reader.tensor("model.embed_tokens.weight")},
    )

    num_experts = config["num_experts"]
    ep_size = args.target_expert_parallel_size
    if num_experts % ep_size:
        raise ValueError(f"{num_experts} experts are not divisible by target EP={ep_size}")

    for layer_idx in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer_idx}"
        query = reader.tensor(f"{prefix}.self_attn.q_proj.weight")
        key = reader.tensor(f"{prefix}.self_attn.k_proj.weight")
        value = reader.tensor(f"{prefix}.self_attn.v_proj.weight")
        message = {
            "input norm weight": reader.tensor(f"{prefix}.input_layernorm.weight"),
            "post norm weight": reader.tensor(f"{prefix}.post_attention_layernorm.weight"),
            "q norm weight": reader.tensor(f"{prefix}.self_attn.q_norm.weight"),
            "k norm weight": reader.tensor(f"{prefix}.self_attn.k_norm.weight"),
            "qkv weight": pack_grouped_query_qkv(
                query,
                key,
                value,
                num_attention_heads=config["num_attention_heads"],
                num_query_groups=config["num_key_value_heads"],
                head_dim=config["head_dim"],
            ),
            "dense weight": reader.tensor(f"{prefix}.self_attn.o_proj.weight"),
            "router weight": reader.tensor(f"{prefix}.mlp.gate.weight"),
        }
        _queue_put(queue, f"transformer layer {layer_idx}", message)

        for ep_rank in range(ep_size):
            global_experts = expert_range_for_ep_rank(num_experts, ep_size, ep_rank)
            gate, up, down = [], [], []
            for expert_idx in global_experts:
                expert = f"{prefix}.mlp.experts.{expert_idx}"
                gate.append(reader.tensor(f"{expert}.gate_proj.weight"))
                up.append(reader.tensor(f"{expert}.up_proj.weight"))
                down.append(reader.tensor(f"{expert}.down_proj.weight"))
            _queue_put(
                queue,
                f"transformer layer {layer_idx} experts {ep_rank}",
                {
                    "mlp l0 weight W": torch.stack(gate),
                    "mlp l0 weight V": torch.stack(up),
                    "mlp l1 weight": torch.stack(down),
                },
            )

    _queue_put(
        queue,
        "final norm",
        {"weight": reader.tensor("model.norm.weight")},
    )
    _queue_put(
        queue,
        "output layer",
        {"weight": reader.tensor("lm_head.weight")},
    )
    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put("exit")
        raise
