"""DSV4 Megatron pretrain helpers (native ``megatron.training.pretrain`` path)."""

from __future__ import annotations

import os
from functools import partial
from typing import Optional

import numpy
import torch

from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from lumen.models.dsv4.megatron.fp8 import dsv4_linear_fp8_enabled, enable_fp8_for_dsv4_model
from lumen.models.dsv4.megatron.pipeline import install_dsv4_pipeline_shape_exchange
from lumen.models.utils import safe_add_argument

__all__ = [
    "add_dsv4_pretrain_args",
    "dsv4_forward_step",
    "dsv4_gpt_builder",
    "dsv4_model_provider",
    "install_dsv4_safe_mock_data",
    "miles_aligned_lm_loss_func",
    "reduce_miles_sample_mean_nll",
]


def install_dsv4_safe_mock_data() -> None:
    """Use safe token ids for Megatron MockGPTDataset.

    Default mock data cycles ``(arange + 1) % vocab_size``, which hits hash-routed
    MoE layers with ``tid2eid[input_ids] == -1`` for unmapped vocab slots and
    triggers GPU faults during ``torch.gather``. GRPO smoke uses small ids
    (e.g. 100+) from fake rollout instead.
    """
    from megatron.core.datasets.gpt_dataset import MockGPTLowLevelDataset

    token_base = int(os.environ.get("DSV4_MOCK_TOKEN_BASE", "100"))

    def _safe_getitem(self, idx: int) -> numpy.number:
        length = self.sequence_lengths[idx]
        # GRPO fake-rollout style ids in [0, 128000); avoid NullTokenizer eod (= vocab_size).
        tokens = numpy.array(
            [(token_base + idx * 7 + j * 13) % 128000 for j in range(length)],
            dtype=numpy.int64,
        )
        return numpy.int64(tokens)

    def _safe_get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        if length is None:
            length = self.sequence_lengths[idx] - offset
        return _safe_getitem(self, idx)[offset : offset + length]

    MockGPTLowLevelDataset.__getitem__ = _safe_getitem  # type: ignore[method-assign]
    MockGPTLowLevelDataset.get = _safe_get  # type: ignore[method-assign]
    print(
        f"DSV4 pretrain: patched MockGPTLowLevelDataset (token_base={token_base}, no eod)",
        flush=True,
    )


def reduce_miles_sample_mean_nll(
    per_token_loss: torch.Tensor,
    loss_mask: torch.Tensor,
    micro_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Miles ``sum_of_sample_mean`` on per-token CE.

    For each of ``micro_batch_size`` samples, take the masked mean NLL, then
    *sum* those sample means. Megatron later divides by the returned sample
    count and ``num_microbatches``, which matches Miles
    ``loss / global_batch_size`` after ``apply_megatron_loss_scaling``.

    When every sample has the same number of unmasked tokens this equals the
    global token-mean CE (Megatron's default logged ``lm loss``).
    """
    losses = per_token_loss.reshape(-1).float()
    mask = loss_mask.reshape(-1).float()
    mbs = max(int(micro_batch_size), 1)
    if losses.numel() == 0:
        zero = losses.sum()
        return zero, torch.tensor(1, device=losses.device, dtype=torch.int)
    if losses.numel() % mbs != 0:
        # Sequence-parallel shards that are not divisible by MBS: fall back to
        # one sample so we still return a valid mean rather than crashing.
        mbs = 1
    token_ce = losses.view(mbs, -1)
    token_mask = mask.view(mbs, -1)
    per_sample = (token_ce * token_mask).sum(dim=-1) / torch.clamp_min(token_mask.sum(dim=-1), 1.0)
    reduced = per_sample.sum()
    num_samples = torch.tensor(mbs, device=losses.device, dtype=torch.int)
    return reduced, num_samples


def miles_aligned_lm_loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    model=None,
):
    """LM loss using Miles' per-sample mean reduction (not global token mean).

    ``output_tensor`` is Megatron per-token CE. Logging uses
    ``report['lm loss'] = [sum_of_sample_means, n_samples]`` so the trainer's
    ``sum / count`` is the Miles metric. Backward uses the same tensor; Megatron
    divides by ``n_samples`` then ``num_microbatches``.
    """
    from megatron.training import get_args

    args = get_args()
    mbs = getattr(args, "micro_batch_size", 1)
    if loss_mask is None or output_tensor is None:
        ref = output_tensor if output_tensor is not None else loss_mask
        device = ref.device if ref is not None else "cpu"
        zero = torch.zeros((), device=device)
        ones = torch.ones((), device=device, dtype=torch.int)
        return zero, ones, {"lm loss": torch.stack([zero.detach(), ones.float()])}

    reduced, num_samples = reduce_miles_sample_mean_nll(output_tensor, loss_mask, mbs)
    report = {
        "lm loss": torch.cat([reduced.detach().view(1), num_samples.float().view(1)])
    }
    return reduced, num_samples, report


def dsv4_forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Forward step for DSV4 Megatron training (``position_ids=None``, no attention mask)."""
    from megatron.core.utils import get_attr_wrapped_model
    from megatron.training import get_args, get_timers
    from pretrain_gpt import get_batch, stimer

    args = get_args()
    timers = get_timers()

    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, _attention_mask, _position_ids, packed_seq_params = get_batch(
            data_iterator, vp_stage
        )
    timers("batch-generator").stop()

    loss_func = miles_aligned_lm_loss_func

    with stimer:
        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm
            schedule_plan = model.build_schedule_plan(
                tokens, None, None, labels=labels, loss_mask=loss_mask
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)
        output_tensor = model(
            tokens,
            None,
            None,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )

    return output_tensor, partial(loss_func, loss_mask, model=model)


def add_dsv4_pretrain_args(parser):
    """Register DSV4-specific Megatron CLI flags."""
    group = parser.add_argument_group(title="dsv4 pretrain")
    safe_add_argument(
        group,
        "--dsv4-dsa-topk-backend",
        type=str,
        default="torch",
        choices=["torch", "flashinfer"],
        help="Top-k backend for DSV4 DSA indexer.",
    )
    safe_add_argument(
        group,
        "--original-max-position-embeddings",
        type=int,
        default=4096,
        help="Original maximum position embeddings for YaRN RoPE (MLA).",
    )
    safe_add_argument(group, "--beta-fast", type=float, default=32, help="YaRN beta_fast (MLA).")
    safe_add_argument(group, "--beta-slow", type=float, default=1, help="YaRN beta_slow (MLA).")
    safe_add_argument(
        group,
        "--moe-router-freeze-gate",
        action="store_true",
        help="Freeze MoE router gate weights during training.",
    )
    safe_add_argument(
        group,
        "--freeze-e-score-correction-bias",
        action="store_true",
        help="Freeze MoE expert score correction bias during training.",
    )
    safe_add_argument(
        group,
        "--no-activation-func-clamp-shared-expert",
        action="store_false",
        dest="activation_func_clamp_shared_expert",
        default=True,
        help="Skip activation clamp inside shared expert MLP.",
    )
    return parser


def dsv4_gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build GPTModel with callable ``--spec`` hooks."""
    print_rank_0("building DSV4 GPT model ...")

    if config is None:
        config = core_transformer_config_from_args(args)
    config.dsv4_mode = True
    # ROCm Megatron's TransformerConfig predates the unpadded vocab_size field
    # used by DSV4 hash routers. tid2eid is [vocab_size, topk] in the checkpoint
    # even though GPT embeddings use padded_vocab_size.
    config.vocab_size = int(args.vocab_size)
    if getattr(args, "pipeline_model_parallel_size", 1) > 1:
        install_dsv4_pipeline_shape_exchange()
        config.variable_seq_lengths = True
        config.batch_p2p_comm = False

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
        if callable(transformer_layer_spec):
            transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
    else:
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_block_with_experimental_attention_variant_spec,
        )

        transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
            config=config, vp_stage=vp_stage
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=None,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )
    return model


def dsv4_model_provider(
    megatron_model_provider,
    gpt_builder,
    pre_process=True,
    post_process=True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
):
    """Build GPTModel via ``dsv4_gpt_builder`` + optional Lumen FP8."""
    model = megatron_model_provider(
        gpt_builder,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
    if dsv4_linear_fp8_enabled():
        enable_fp8_for_dsv4_model(model)
    if os.environ.get("DSV4_DUMP_LAYER_ACT", "0") == "1":
        from lumen.models.dsv4.megatron.act_dump import install_layer_act_dump

        install_layer_act_dump(model, tag=os.environ.get("DSV4_DUMP_TAG", "lumen"))
    if os.environ.get("DSV4_DUMP_MHC_DX", "0") == "1":
        from lumen.models.dsv4.megatron.act_dump import install_mhc_dx_ln_hooks

        install_mhc_dx_ln_hooks(model)
    return model
