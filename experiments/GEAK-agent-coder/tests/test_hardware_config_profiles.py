import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from geak_agent_coder.config import (
    assert_runtime_compatible,
    validate_training_config,
)
from geak_agent_coder.sft.launcher import build_torchrun_command, load_config
from geak_agent_coder.sft.train import _lumen_args


ROOT = Path(__file__).parents[1]
MAIN_CONFIG = ROOT / "configs/sft/mi308_ep8_fp8.yaml"


def test_main_config_composes_hardware_and_strict_recipe():
    config = load_config(MAIN_CONFIG)

    assert config["hardware"]["profile"] == "amd_mi308_gfx942"
    assert config["recipe"]["name"] == "geak_mi308_release_v1"
    assert config["distributed"]["world_size"] == 8
    assert config["precision"]["fp8_format"] == "e4m3fnuz"
    assert [item["role"] for item in config["_composition"]["profiles"]] == [
        "hardware",
        "recipe",
    ]
    command = build_torchrun_command(MAIN_CONFIG, python_executable="python")
    assert "--nnodes=1" in command
    assert "--nproc-per-node=8" in command
    assert "--standalone" in command


def test_generic_validation_uses_world_and_qwen_expert_topology():
    config = load_config(MAIN_CONFIG)
    config["recipe"]["name"] = "custom"
    config["hardware"]["gpus_per_node"] = 4
    config["distributed"].update(
        nproc_per_node=4,
        world_size=4,
        dp_size=4,
        ep_size=4,
    )
    config["lora"].update(
        attention_rank=12,
        attention_alpha=24,
        expert_rank=3,
        expert_alpha=6,
    )
    validate_training_config(config)

    invalid = copy.deepcopy(config)
    invalid["distributed"]["ep_size"] = 3
    with pytest.raises(ValueError, match="must divide world_size"):
        validate_training_config(invalid)

    invalid = copy.deepcopy(config)
    invalid["distributed"].update(ep_size=6, world_size=6, dp_size=6)
    invalid["hardware"]["gpus_per_node"] = 6
    invalid["distributed"]["nproc_per_node"] = 6
    with pytest.raises(ValueError, match="128 experts"):
        validate_training_config(invalid)


def test_strict_release_rejects_parameter_drift():
    config = load_config(MAIN_CONFIG)
    config["lora"]["expert_rank"] = 4
    with pytest.raises(ValueError, match="immutable"):
        validate_training_config(config)


def test_fp8_capability_mismatch_has_no_bf16_fallback():
    config = load_config(MAIN_CONFIG)
    config["recipe"]["name"] = "custom"
    config["hardware"]["capabilities"]["native_fp8"] = False
    config["hardware"]["capabilities"]["fp8_formats"] = []
    with pytest.raises(ValueError, match="explicit BF16"):
        validate_training_config(config)


class _FakeCuda:
    def __init__(self, architecture):
        self.architecture = architecture

    def is_available(self):
        return True

    def get_device_properties(self, _index):
        return SimpleNamespace(gcnArchName=self.architecture)


def _fake_torch(architecture):
    return SimpleNamespace(
        version=SimpleNamespace(hip="7.0"),
        cuda=_FakeCuda(architecture),
    )


def test_runtime_arch_check_happens_against_declared_profile():
    config = load_config(MAIN_CONFIG)
    assert (
        assert_runtime_compatible(config, torch_module=_fake_torch("gfx942:sramecc+"))
        == "gfx942"
    )
    with pytest.raises(RuntimeError, match="does not match"):
        assert_runtime_compatible(config, torch_module=_fake_torch("gfx950"))


def test_lumen_args_read_precision_scaling_and_block_size():
    config = load_config(MAIN_CONFIG)
    config["recipe"]["name"] = "custom"
    config["precision"]["scaling"] = "custom_scaling"
    config["precision"]["block_size"] = 64
    args = _lumen_args(config)
    assert args.mode == "fp8_blockwise2d"
    assert args.fp8_scaling == "custom_scaling"
    assert args.fp8_block_size == 64
    assert args.fp8_format == "e4m3fnuz"


def test_flat_config_remains_loadable_with_migration_warning(tmp_path):
    # A tiny self-contained legacy config exercises the old field names.
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """
model: {name_or_path: Qwen/Qwen3-30B-A3B}
distributed:
  nproc_per_node: 2
  ep_size: 2
  expert_backend: sequential
lora:
  attention_rank: 7
  attention_alpha: 14
  expert_rank: 5
  expert_alpha: 10
precision:
  mode: fp8_blockwise2d
  base_scaling: blockwise2d
  block_size: 128
""",
        encoding="utf-8",
    )
    with pytest.warns(FutureWarning, match="migrate"):
        loaded = load_config(path)
    assert loaded["distributed"]["nproc_per_node"] == 2
    assert "hardware" not in loaded
