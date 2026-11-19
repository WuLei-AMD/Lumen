"""Save Qwen3-MoE converter messages with the bundled MCore 0.17 saver."""

from __future__ import annotations

import sys
from functools import partial

import torch

from qwen3_moe_mapping import pack_swiglu_fc1
from saver_base import MegatronCheckpointSaverBase
from schema_qwen3_moe import get_qwen3_moe_schema
from utils import _ConverterFakeProcessGroup, chunk_weight


def add_arguments(parser):
    group = parser.add_argument_group(title="Qwen3-MoE MCore saver")
    group.add_argument("--megatron-path", type=str, default=None)
    group.add_argument("--target-tensor-parallel-size", type=int, default=1)
    group.add_argument("--target-pipeline-parallel-size", type=int, default=1)
    group.add_argument("--target-expert-parallel-size", type=int, default=8)
    group.add_argument(
        "--saver-transformer-impl",
        default="transformer_engine",
        choices=["transformer_engine"],
    )


class Qwen3MoECheckpointSaver(MegatronCheckpointSaverBase):
    def initialize_megatron_env(self):
        super().initialize_megatron_env()
        # MCore 0.17 expert MLP construction requires the ETP group itself,
        # while the bundled generic converter only initializes TP/PP/EP groups.
        from megatron.core import mpu

        expert_tp_size = self.margs.expert_tensor_parallel_size
        mpu._EXPERT_TENSOR_PARALLEL_GROUP = _ConverterFakeProcessGroup(
            size=expert_tp_size
        )
        mpu.set_expert_tensor_parallel_world_size(expert_tp_size)
        mpu.set_expert_tensor_parallel_rank(0)

    def receive_checkpoint_metadata(self):
        super().receive_checkpoint_metadata()
        actual = (
            self.args.target_tensor_parallel_size,
            self.args.target_pipeline_parallel_size,
            self.args.target_expert_parallel_size,
        )
        if actual != (1, 1, 8):
            raise ValueError(f"this converter targets TP=1, PP=1, EP=8; got {actual}")
        if self.md.num_experts != 128:
            raise ValueError(f"expected 128 experts, got {self.md.num_experts}")

    def _maybe_parse_additional_megatron_args(self, margs):
        margs.sequence_parallel = self.args.target_expert_parallel_size > 1
        margs.qk_layernorm = True
        margs.group_query_attention = True
        margs.num_query_groups = self.md.num_query_groups
        margs.kv_channels = self.md.kv_channels
        margs.moe_grouped_gemm = False
        margs.expert_tensor_parallel_size = 1
        return margs

    def import_model_provider(self):
        try:
            from megatron.core.enums import ModelType
            from gpt_builders import gpt_builder
            from model_provider import model_provider
        except ModuleNotFoundError as error:
            print(f"Unable to import required Megatron modules: {error}")
            sys.exit(1)
        self.model_provider = partial(model_provider, gpt_builder)
        self.margs.model_type = ModelType.encoder_or_decoder

    @staticmethod
    def _tp_expert_weight(weight: torch.Tensor, mode: str, tp_size: int):
        # The loader already sends one EP rank's contiguous local expert stack.
        return chunk_weight(weight, mode, tp_size, 1)[0]

    def receive_model(self):
        schema = get_qwen3_moe_schema(
            self.md.num_experts // self.args.target_expert_parallel_size,
        )
        self._receive_qwen3_lm(schema)

    def _receive_qwen3_lm(self, schema):
        from megatron.core import mpu
        from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding

        embeddings_msg = self.queue_get("embeddings")
        word_embeddings = embeddings_msg.pop("word embeddings")
        self.check_message(embeddings_msg)
        self.margs.padded_vocab_size = vocab_size_with_padding(
            self.md.true_vocab_size, self.margs
        )
        if self.margs.padded_vocab_size != word_embeddings.shape[0]:
            raise ValueError(
                f"target padded vocab is {self.margs.padded_vocab_size}, but HF matrix has "
                f"{word_embeddings.shape[0]} rows"
            )
        out_word_embeddings = torch.chunk(
            word_embeddings, self.args.target_tensor_parallel_size, dim=0
        )
        for ep_rank in range(self.args.target_expert_parallel_size):
            for tp_rank in range(self.args.target_tensor_parallel_size):
                schema.set(
                    "embeddings",
                    self.get_local_model(0, ep_rank, tp_rank),
                    {"pos": None, "word": out_word_embeddings[tp_rank]},
                )

        total_layer_num = 0
        for pp_rank in range(self.args.target_pipeline_parallel_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            self.get_local_model(pp_rank, 0, 0)
            num_local_layers = schema.get_num_layers(self.models[pp_rank][0][0])
            for layer_id in range(num_local_layers):
                name = f"transformer layer {total_layer_num}"
                msg = self.queue_get(name)
                input_norm = msg.pop("input norm weight")
                post_norm = msg.pop("post norm weight")
                q_norm = msg.pop("q norm weight")
                k_norm = msg.pop("k norm weight")
                router = msg.pop("router weight")
                qkv = chunk_weight(
                    msg.pop("qkv weight"),
                    "column",
                    self.args.target_tensor_parallel_size,
                )
                dense = chunk_weight(
                    msg.pop("dense weight"),
                    "row",
                    self.args.target_tensor_parallel_size,
                )

                self.check_message(msg)

                for ep_rank in range(self.args.target_expert_parallel_size):
                    for tp_rank in range(self.args.target_tensor_parallel_size):
                        params = {
                            "self_attn_norm_weight": input_norm,
                            "self_attn_qkv_weight": qkv[tp_rank],
                            "self_attn_proj_weight": dense[tp_rank],
                            "q_norm_weight": q_norm,
                            "k_norm_weight": k_norm,
                            "mlp_norm_weight": post_norm,
                            "router_weight": router,
                        }
                        schema.set_layer(
                            self.get_local_model(pp_rank, ep_rank, tp_rank),
                            layer_id,
                            params,
                        )

                for ep_rank in range(self.args.target_expert_parallel_size):
                    expert_msg = self.queue_get(f"{name} experts {ep_rank}")
                    gate = self._tp_expert_weight(
                        expert_msg.pop("mlp l0 weight W"),
                        "column",
                        self.args.target_tensor_parallel_size,
                    )
                    up = self._tp_expert_weight(
                        expert_msg.pop("mlp l0 weight V"),
                        "column",
                        self.args.target_tensor_parallel_size,
                    )
                    down = self._tp_expert_weight(
                        expert_msg.pop("mlp l1 weight"),
                        "row",
                        self.args.target_tensor_parallel_size,
                    )
                    self.check_message(expert_msg)
                    for tp_rank in range(self.args.target_tensor_parallel_size):
                        local_fc1 = pack_swiglu_fc1(gate[tp_rank], up[tp_rank])
                        params = {}
                        for local_expert_idx in range(local_fc1.shape[0]):
                            params[f"mlp_fc1_weight.{local_expert_idx}"] = local_fc1[
                                local_expert_idx
                            ]
                            params[f"mlp_fc2_weight.{local_expert_idx}"] = down[tp_rank][
                                local_expert_idx
                            ]
                        schema.set_layer(
                            self.get_local_model(pp_rank, ep_rank, tp_rank),
                            layer_id,
                            params,
                        )
                total_layer_num += 1

        final_msg = self.queue_get("final norm")
        final_norm = final_msg.pop("weight")
        self.check_message(final_msg)
        output_msg = self.queue_get("output layer")
        output = output_msg.pop("weight")
        self.check_message(output_msg)
        output = torch.chunk(output, self.args.target_tensor_parallel_size, dim=0)

        last_pp = self.args.target_pipeline_parallel_size - 1
        for ep_rank in range(self.args.target_expert_parallel_size):
            for tp_rank in range(self.args.target_tensor_parallel_size):
                model = self.get_local_model(last_pp, ep_rank, tp_rank)
                schema.set("final_norm", model, {"weight": final_norm, "bias": None})
                schema.set("output_layer", model, {"weight": output[tp_rank]})

        if self.queue_get() != "done":
            raise RuntimeError("expected final 'done' message")


def save_checkpoint(queue, args):
    Qwen3MoECheckpointSaver(args, queue).save()
