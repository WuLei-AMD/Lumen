"""Composable training configuration and AMD runtime validation."""

from __future__ import annotations

import copy
import fnmatch
import warnings
from pathlib import Path
from typing import Any, Mapping

QWEN3_MODEL = "Qwen/Qwen3-30B-A3B"
QWEN3_CODER_MODELS = {
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
}
QWEN3_EXPERTS = 128
PROFILE_KEY = "profiles"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("loading SFT config requires PyYAML") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"config {path} must contain a mapping")
    return value


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _profile_paths(value: object) -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        entries = list(value.items())
    elif isinstance(value, list):
        entries = [(str(index), item) for index, item in enumerate(value)]
    else:
        raise ValueError("profiles must be a mapping or ordered list of YAML paths")
    paths: list[tuple[str, str]] = []
    for role, path in entries:
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"profile {role!r} must reference a YAML path")
        paths.append((str(role), path))
    return paths


def load_composed_config(path: str | Path) -> dict[str, Any]:
    """Load profile overlays in order, then apply the main file as overrides."""

    main_path = Path(path).resolve()
    main = _load_yaml(main_path)
    references = main.pop(PROFILE_KEY, None)
    if references is None:
        warnings.warn(
            "flat SFT YAML is supported for compatibility; migrate by adding "
            "'profiles: {hardware: ..., recipe: ...}'",
            FutureWarning,
            stacklevel=2,
        )
        return main

    composed: dict[str, Any] = {}
    loaded_profiles: list[dict[str, str]] = []
    for role, reference in _profile_paths(references):
        profile_path = (main_path.parent / reference).resolve()
        profile = _load_yaml(profile_path)
        if PROFILE_KEY in profile:
            raise ValueError(
                f"nested profiles are not supported ({profile_path}); "
                "reference every profile from the main training YAML"
            )
        composed = _deep_merge(composed, profile)
        loaded_profiles.append(
            {"role": role, "path": str(profile_path), "name": profile_path.stem}
        )
    composed = _deep_merge(composed, main)
    composed["_composition"] = {"profiles": loaded_profiles, "main": str(main_path)}
    return composed


def _positive_int(section: Mapping[str, Any], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _validate_topology(config: Mapping[str, Any]) -> None:
    distributed = config.get("distributed", {})
    hardware = config.get("hardware", {})
    if not isinstance(distributed, Mapping) or not isinstance(hardware, Mapping):
        raise ValueError("distributed and hardware must be mappings")

    nproc = _positive_int(distributed, "nproc_per_node")
    nodes = hardware.get("nodes", 1)
    if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes <= 0:
        raise ValueError("hardware.nodes must be a positive integer")
    expected_world = nodes * nproc
    world_size = distributed.get("world_size", expected_world)
    if world_size != expected_world:
        raise ValueError(
            f"distributed.world_size={world_size} must equal "
            f"hardware.nodes*nproc_per_node={expected_world}"
        )
    gpus_per_node = hardware.get("gpus_per_node")
    if gpus_per_node is not None and gpus_per_node != nproc:
        raise ValueError(
            "hardware.gpus_per_node must equal distributed.nproc_per_node"
        )
    dp_size = distributed.get("dp_size", world_size)
    if dp_size != world_size:
        raise ValueError(
            "distributed.dp_size must equal world_size because dense DP overlaps EP"
        )
    ep_size = _positive_int(distributed, "ep_size")
    if world_size % ep_size:
        raise ValueError(f"ep_size={ep_size} must divide world_size={world_size}")
    if QWEN3_EXPERTS % ep_size:
        raise ValueError(
            f"ep_size={ep_size} must divide Qwen3's {QWEN3_EXPERTS} experts"
        )


def _validate_lora(config: Mapping[str, Any]) -> None:
    lora = config.get("lora", {})
    if not isinstance(lora, Mapping):
        raise ValueError("lora must be a mapping")
    for key in ("attention_rank", "expert_rank"):
        _positive_int(lora, key)
    for key in ("attention_alpha", "expert_alpha"):
        value = lora.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"lora.{key} must be positive")
    targets = lora.get("targets")
    if targets is not None:
        if not isinstance(targets, Mapping):
            raise ValueError("lora.targets must be a mapping")
        expected = {
            "attention": {"q_proj", "k_proj", "v_proj", "o_proj"},
            "experts": {"gate_up_proj", "down_proj"},
        }
        for family, supported in expected.items():
            configured = targets.get(family)
            if not isinstance(configured, list) or not configured:
                raise ValueError(f"lora.targets.{family} must be a non-empty list")
            unknown = set(configured) - supported
            if unknown:
                raise ValueError(
                    f"unsupported Qwen3 {family} LoRA targets: {sorted(unknown)}"
                )


def _validate_precision(config: Mapping[str, Any]) -> None:
    precision = config.get("precision", {})
    hardware = config.get("hardware", {})
    capabilities = hardware.get("capabilities", {}) if isinstance(hardware, Mapping) else {}
    if not isinstance(precision, Mapping) or not isinstance(capabilities, Mapping):
        raise ValueError("precision and hardware.capabilities must be mappings")
    mode = precision.get("mode")
    if mode not in {"bf16", "fp8_blockwise2d"}:
        raise ValueError("precision.mode must be 'bf16' or 'fp8_blockwise2d'")
    if precision.get("adapter_dtype", "bfloat16") != "bfloat16":
        raise ValueError("hierarchical LoRA currently requires BF16 adapters")
    if mode == "bf16":
        if capabilities and capabilities.get("bf16") is not True:
            raise ValueError("hardware profile does not explicitly support BF16")
        return

    if not hardware:
        scaling = precision.get("scaling", precision.get("base_scaling"))
        if not isinstance(scaling, str) or not scaling:
            raise ValueError("legacy FP8 config requires precision.base_scaling")
        block_size = precision.get("block_size")
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("legacy FP8 config requires precision.block_size")
        return

    if capabilities.get("native_fp8") is not True:
        raise ValueError(
            "FP8 was requested but hardware.capabilities.native_fp8 is not true; "
            "select an explicit BF16 recipe/profile instead"
        )
    fp8_format = precision.get("fp8_format")
    formats = capabilities.get("fp8_formats")
    if not isinstance(formats, list) or fp8_format not in formats:
        raise ValueError(
            f"precision.fp8_format={fp8_format!r} is not supported by "
            f"hardware profile formats={formats!r}"
        )
    scaling = precision.get("scaling", precision.get("base_scaling"))
    if not isinstance(scaling, str) or not scaling:
        raise ValueError("FP8 precision requires precision.scaling")
    block_size = precision.get("block_size")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("FP8 precision requires a positive precision.block_size")


def _validate_strict_release(config: Mapping[str, Any]) -> None:
    recipe = config.get("recipe", {})
    if not isinstance(recipe, Mapping) or recipe.get("name") != "geak_mi308_release_v1":
        return
    hardware = config.get("hardware", {})
    distributed = config["distributed"]
    lora = config["lora"]
    precision = config["precision"]
    required = {
        "hardware.profile": (hardware.get("profile"), "amd_mi308_gfx942"),
        "hardware.gpu_sku": (hardware.get("gpu_sku"), "MI308X"),
        "hardware.target_arch": (hardware.get("target_arch"), "gfx942"),
        "distributed.nproc_per_node": (distributed.get("nproc_per_node"), 8),
        "distributed.world_size": (distributed.get("world_size"), 8),
        "distributed.dp_size": (distributed.get("dp_size"), 8),
        "distributed.ep_size": (distributed.get("ep_size"), 8),
        "distributed.expert_backend": (
            distributed.get("expert_backend"),
            "sonic",
        ),
        "lora.attention_rank": (lora.get("attention_rank"), 32),
        "lora.attention_alpha": (lora.get("attention_alpha"), 64),
        "lora.expert_rank": (lora.get("expert_rank"), 8),
        "lora.expert_alpha": (lora.get("expert_alpha"), 16),
        "precision.mode": (precision.get("mode"), "fp8_blockwise2d"),
        "precision.fp8_format": (precision.get("fp8_format"), "e4m3fnuz"),
        "precision.scaling": (
            precision.get("scaling", precision.get("base_scaling")),
            "blockwise2d",
        ),
        "precision.block_size": (precision.get("block_size"), 128),
    }
    mismatches = {
        key: {"actual": actual, "required": expected}
        for key, (actual, expected) in required.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(
            "geak_mi308_release_v1 is immutable; mismatches=" + repr(mismatches)
        )


def validate_training_config(config: Mapping[str, Any]) -> None:
    model = config.get("model", {})
    if not isinstance(model, Mapping) or model.get("name_or_path") not in QWEN3_CODER_MODELS:
        raise ValueError(
            f"this training backend supports: {sorted(QWEN3_CODER_MODELS)}"
        )
    _validate_topology(config)
    _validate_lora(config)
    distributed = config["distributed"]
    if distributed.get("expert_backend") not in {"sequential", "sonic"}:
        raise ValueError(
            "Qwen3 expert LoRA requires distributed.expert_backend to expose "
            "sequential or packed Sonic expert weights"
        )
    _validate_precision(config)
    _validate_strict_release(config)


def normalize_architecture(value: str) -> str:
    return value.strip().lower().split(":", 1)[0]


def detect_runtime_arch(torch_module=None, *, device_index: int = 0) -> str:
    """Return the active ROCm GCN architecture without initializing distributed."""

    if torch_module is None:
        import torch as torch_module
    hip_version = getattr(getattr(torch_module, "version", None), "hip", None)
    if not hip_version:
        raise RuntimeError("AMD hardware profiles require a ROCm-enabled PyTorch build")
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise RuntimeError("ROCm reports no available GPU")
    properties = cuda.get_device_properties(device_index)
    architecture = getattr(properties, "gcnArchName", None)
    if not isinstance(architecture, str) or not architecture:
        raise RuntimeError("unable to detect GPU gcnArchName from PyTorch")
    return normalize_architecture(architecture)


def assert_runtime_compatible(
    config: Mapping[str, Any],
    *,
    torch_module=None,
    device_index: int = 0,
) -> str:
    """Fail before process-group creation when runtime and profile disagree."""

    hardware = config.get("hardware", {})
    if not isinstance(hardware, Mapping) or not hardware:
        return "unchecked-legacy-flat-config"
    if hardware.get("runtime_arch_check", True) is not True:
        raise ValueError("hardware.runtime_arch_check cannot be disabled")
    actual = detect_runtime_arch(torch_module, device_index=device_index)
    target = hardware.get("target_arch")
    supported = hardware.get("supported_arches")
    patterns = supported if isinstance(supported, list) else [target]
    if not patterns or any(not isinstance(pattern, str) for pattern in patterns):
        raise ValueError("hardware profile must declare target_arch or supported_arches")
    if not any(fnmatch.fnmatch(actual, pattern.lower()) for pattern in patterns):
        raise RuntimeError(
            f"runtime GPU architecture {actual!r} does not match declared "
            f"hardware profile {patterns!r}"
        )
    return actual


def validate_launch_environment(config: Mapping[str, Any], environ: Mapping[str, str]) -> None:
    """Check torchrun topology variables before distributed initialization."""

    distributed = config["distributed"]
    hardware = config.get("hardware", {})
    nodes = hardware.get("nodes", 1) if isinstance(hardware, Mapping) else 1
    expected_world = distributed.get(
        "world_size", nodes * distributed["nproc_per_node"]
    )
    expected_local = distributed["nproc_per_node"]
    for name, expected in (
        ("WORLD_SIZE", expected_world),
        ("LOCAL_WORLD_SIZE", expected_local),
    ):
        value = environ.get(name)
        if value is not None and int(value) != expected:
            raise RuntimeError(
                f"torchrun {name}={value} does not match configured {expected}"
            )
