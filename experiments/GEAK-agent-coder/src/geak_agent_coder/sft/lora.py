"""Small, MoE-aware LoRA implementation used by the SFT entry point."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


ATTENTION_PROJECTIONS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
EXPERT_PROJECTIONS = frozenset({"gate_up_proj", "down_proj"})


@dataclass(frozen=True)
class LoRAConfig:
    attention_rank: int = 32
    attention_alpha: float = 64.0
    expert_rank: int = 8
    expert_alpha: float = 16.0
    dropout: float = 0.0
    dtype: str = "bfloat16"
    attention_targets: tuple[str, ...] = tuple(sorted(ATTENTION_PROJECTIONS))
    expert_targets: tuple[str, ...] = tuple(sorted(EXPERT_PROJECTIONS))


@dataclass(frozen=True)
class TrainableAudit:
    total_parameters: int
    trainable_parameters: int
    trainable_names: tuple[str, ...]
    attention_parameters: int
    expert_parameters: int


class LoRALinear(nn.Module):
    """BF16 low-rank update around a patchable base ``nn.Linear``."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRA requires nn.Linear, got {type(base_layer).__name__}")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if dtype is None:
            dtype = torch.bfloat16
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        device = base_layer.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base_layer.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_layer.out_features, self.rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        # Adapters stay BF16 even if Lumen replaces base_layer with an FP8
        # compute module. Cast only at the adapter boundary.
        adapter_input = self.dropout(inputs).to(self.lora_A.dtype)
        update = (adapter_input @ self.lora_A.T) @ self.lora_B.T
        return base + update.to(base.dtype) * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha:g}"


class PackedExpertLoRA(nn.Module):
    """Per-expert low-rank delta for packed grouped-GEMM weights."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(
            torch.empty(
                num_experts,
                self.rank,
                in_features,
                dtype=dtype,
                device=device,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                num_experts,
                out_features,
                self.rank,
                dtype=dtype,
                device=device,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def delta(self) -> torch.Tensor:
        return (
            torch.bmm(self.lora_A.transpose(1, 2), self.lora_B.transpose(1, 2))
            * self.scaling
        )


def validate_expert_backend(expert_backend: str, *, expert_lora: bool = True) -> None:
    """Fail closed when expert LoRA cannot target individual local linears."""
    supported = {"sequential", "sonic"}
    known = supported | {"sonic", "te_grouped"}
    if expert_backend not in known:
        raise ValueError(f"unknown expert backend {expert_backend!r}")
    if expert_lora and expert_backend not in supported:
        raise ValueError(
            "expert LoRA requires expert_backend='sequential'; "
            f"{expert_backend!r} does not expose per-expert nn.Linear modules"
        )


def _adapter_dtype(name: str):
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unsupported adapter dtype {name!r}") from exc


def inject_hierarchical_lora(
    model: nn.Module,
    config: LoRAConfig = LoRAConfig(),
    *,
    expert_backend: str = "sequential",
) -> tuple[str, ...]:
    """Freeze ``model`` and inject attention/expert adapters in-place.

    This must be called after Lumen's ``shard_moe_experts``. Expert matching is
    deliberately restricted to paths below ``local_experts`` so packed global
    expert tensors, routers, shared MLPs, embeddings, norms and lm_head cannot
    accidentally become trainable.
    """
    validate_expert_backend(expert_backend)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    replacements: list[tuple[nn.Module, str, nn.Linear, int, float, str]] = []
    for path, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{path}.{child_name}" if path else child_name
            if child_name in config.attention_targets and "local_experts" not in full_name:
                replacements.append(
                    (
                        module,
                        child_name,
                        child,
                        config.attention_rank,
                        config.attention_alpha,
                        full_name,
                    )
                )
            elif (
                child_name in config.expert_targets
                and ".local_experts." in f".{full_name}."
            ):
                replacements.append(
                    (
                        module,
                        child_name,
                        child,
                        config.expert_rank,
                        config.expert_alpha,
                        full_name,
                    )
                )

    dtype = _adapter_dtype(config.dtype)
    injected = []
    for parent, name, base, rank, alpha, full_name in replacements:
        setattr(parent, name, LoRALinear(base, rank, alpha, config.dropout, dtype))
        injected.append(full_name)
    if expert_backend == "sonic":
        for path, module in model.named_modules():
            if not path.endswith("local_experts"):
                continue
            w1 = getattr(module, "w1", None)
            w2 = getattr(module, "w2", None)
            if not isinstance(w1, nn.Parameter) or not isinstance(w2, nn.Parameter):
                continue
            if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]:
                raise RuntimeError(f"{path}: unsupported Sonic packed expert layout")
            module.gate_up_lora = PackedExpertLoRA(
                int(w1.shape[0]),
                int(w1.shape[1]),
                int(w1.shape[2]),
                config.expert_rank,
                config.expert_alpha,
                dtype,
                w1.device,
            )
            module.down_lora = PackedExpertLoRA(
                int(w2.shape[0]),
                int(w2.shape[1]),
                int(w2.shape[2]),
                config.expert_rank,
                config.expert_alpha,
                dtype,
                w2.device,
            )
            injected.extend(
                (
                    f"{path}.gate_up_lora",
                    f"{path}.down_lora",
                )
            )
    if not injected:
        raise RuntimeError(
            "no LoRA targets found; call shard_moe_experts before adapter injection"
        )
    if not any(
        name.rsplit(".", 1)[-1] in config.expert_targets
        or name.endswith((".gate_up_lora", ".down_lora"))
        for name in injected
    ):
        raise RuntimeError(
            "no local expert LoRA targets found; expected sequential sharded experts"
        )
    return tuple(injected)


def audit_trainable_parameters(model: nn.Module, *, fail: bool = True) -> TrainableAudit:
    """Verify that every and only LoRA A/B tensors are trainable BF16 tensors."""
    total = trainable = attention = expert = 0
    names: list[str] = []
    violations: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if not parameter.requires_grad:
            continue
        trainable += count
        names.append(name)
        if not (name.endswith(".lora_A") or name.endswith(".lora_B")):
            violations.append(f"non-adapter parameter is trainable: {name}")
        if parameter.dtype != torch.bfloat16:
            violations.append(f"adapter is not BF16: {name} ({parameter.dtype})")
        if ".local_experts." in f".{name}.":
            expert += count
        else:
            attention += count
    if trainable == 0:
        violations.append("model has no trainable adapter parameters")
    if attention == 0:
        violations.append("model has no trainable attention adapters")
    if expert == 0:
        violations.append("model has no trainable expert adapters")
    if fail and violations:
        raise RuntimeError("trainable parameter audit failed: " + "; ".join(violations))
    return TrainableAudit(total, trainable, tuple(names), attention, expert)
