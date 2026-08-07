from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
MODELS_DSV4_INIT_PATH = REPO_ROOT / "lumen/models/dsv4/__init__.py"
CANONICAL_OPS_INIT_PATH = REPO_ROOT / "lumen/ops/dsv4/__init__.py"
COMPRESSOR_PATH = REPO_ROOT / "lumen/models/dsv4/compressor.py"
HYPER_CONNECTION_PATH = REPO_ROOT / "lumen/models/dsv4/hyper_connection.py"
PRETRAIN_PATH = REPO_ROOT / "lumen/models/dsv4/megatron/pretrain.py"
PIPELINE_PATH = REPO_ROOT / "lumen/models/dsv4/megatron/pipeline.py"
DSV4_EXAMPLES = REPO_ROOT / "examples/dsv4"

CANONICAL_OPS = {
    "sparse_mla.py",
    "indexer.py",
    "indexer_utils.py",
    "cp_utils.py",
    "dsa_topk.py",
    "rope.py",
    "qat.py",
    "utils.py",
}
CANONICAL_KERNELS = {
    "sparse_mla": {
        "triton_sparse_mla.py",
        "tilelang_sparse_mla.py",
        "tilelang_sparse_mla_fwd.py",
        "tilelang_sparse_mla_bwd.py",
    },
    "indexer": {
        "aiter_dsv4_indexer.py",
        "tilelang_indexer.py",
        "tilelang_indexer_fwd.py",
        "tilelang_indexer_bwd.py",
    },
    "quant": {"precision_aligned_ops.py", "act_quant.py"},
}


def test_dsv4_sources_live_in_canonical_ops_and_kernel_trees():
    canonical_ops = REPO_ROOT / "lumen/ops/dsv4"
    for filename in CANONICAL_OPS:
        assert (canonical_ops / filename).is_file(), filename

    canonical_kernels = REPO_ROOT / "lumen/kernels/dsv4"
    for group, filenames in CANONICAL_KERNELS.items():
        assert (canonical_kernels / group / "__init__.py").is_file(), group
        for filename in filenames:
            assert (canonical_kernels / group / filename).is_file(), filename

    assert not (REPO_ROOT / "lumen/models/dsv4/ops").exists()
    assert not (REPO_ROOT / "lumen/models/dsv4/modules").exists()
    assert not (REPO_ROOT / "lumen/models/dsv4/tools").exists()
    assert not (REPO_ROOT / "lumen/dsv4").exists()
    assert (COMPRESSOR_PATH).is_file()
    assert (HYPER_CONNECTION_PATH).is_file()
    assert (MODELS_DSV4_INIT_PATH).is_file()


def test_internal_python_imports_use_canonical_dsv4_paths():
    forbidden = (
        "lumen.models.dsv4.ops.kernel",
        "lumen.models.dsv4.ops.sparse_mla_backend",
        "lumen.models.dsv4.ops.indexer_backend",
        "lumen.models.dsv4.ops.indexer_utils",
        "lumen.models.dsv4.ops.cp_utils",
        "lumen.models.dsv4.ops.dsa_topk",
        "lumen.models.dsv4.ops.rope",
        "lumen.models.dsv4.ops.qat",
        "lumen.models.dsv4.ops.utils",
        "lumen.models.dsv4.ops.compressor",
        "lumen.models.dsv4.ops.hyper_connection",
        "lumen.models.dsv4.modules.compressor",
        "lumen.models.dsv4.modules.hyper_connection",
        "lumen.models.dsv4.tools.gen_fake_rollout_data",
        "lumen.dsv4.modules.compressor",
        "lumen.dsv4.modules.hyper_connection",
        "lumen.dsv4.tools.gen_fake_rollout_data",
    )
    patch_hc = DSV4_EXAMPLES / "megatron_patch/hc.py"
    for root in (REPO_ROOT / "lumen", REPO_ROOT / "examples"):
        for path in root.rglob("*.py"):
            if ".bootstrap-build" in path.parts:
                continue
            if path == patch_hc:
                continue
            source = path.read_text(encoding="utf-8")
            assert not any(old_path in source for old_path in forbidden), path


def test_canonical_ops_facade_does_not_eagerly_import_qat_or_kernels():
    tree = ast.parse(CANONICAL_OPS_INIT_PATH.read_text(encoding="utf-8"))
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not isinstance(node, ast.ImportFrom)
        or not (node.module or "").startswith(
            ("lumen.ops.dsv4.qat", "lumen.kernels.dsv4")
        )
        for node in imports
    ), CANONICAL_OPS_INIT_PATH


def test_canonical_dsv4_modules_import_without_model_or_tilekernels(monkeypatch):
    lumen_package = types.ModuleType("lumen")
    lumen_package.__path__ = [str(REPO_ROOT / "lumen")]
    ops_package = types.ModuleType("lumen.ops")
    ops_package.__path__ = [str(REPO_ROOT / "lumen/ops")]
    monkeypatch.setitem(sys.modules, "lumen", lumen_package)
    monkeypatch.setitem(sys.modules, "lumen.ops", ops_package)

    transformer = types.ModuleType("megatron.core.transformer")
    transformer.TransformerConfig = object
    for name in ("megatron", "megatron.core"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, transformer.__name__, transformer)

    for name in tuple(sys.modules):
        if name.startswith(
            (
                "lumen.ops.dsv4",
                "lumen.models.dsv4",
                "lumen.kernels.dsv4",
                "tile_kernels",
            )
        ):
            monkeypatch.delitem(sys.modules, name)

    module_names = (
        "cp_utils",
        "dsa_topk",
        "indexer",
        "indexer_utils",
        "qat",
        "rope",
        "sparse_mla",
        "utils",
    )
    for module_name in module_names:
        module = importlib.import_module(f"lumen.ops.dsv4.{module_name}")
        assert Path(module.__file__).is_relative_to(REPO_ROOT / "lumen/ops/dsv4")

    assert not any(name.startswith("lumen.models.dsv4") for name in sys.modules)
    assert not any(name.startswith("lumen.kernels.dsv4") for name in sys.modules)
    assert not any(name.startswith("tile_kernels") for name in sys.modules)


def test_indexer_is_fixed_to_aiter(monkeypatch):
    indexer = importlib.import_module("lumen.ops.dsv4.indexer")
    aiter_marker = object()

    aiter_module = types.ModuleType(
        "lumen.kernels.dsv4.indexer.aiter_dsv4_indexer"
    )
    aiter_module.batched_indexer_fwd = aiter_marker
    aiter_module.batched_indexer_bwd = aiter_marker
    monkeypatch.setitem(sys.modules, aiter_module.__name__, aiter_module)

    monkeypatch.setenv("V4_INDEXER_IMPL", "tilelang")
    assert indexer.get_indexer_impl() == "aiter"
    assert indexer.get_batched_indexer_fwd() is aiter_marker
    assert indexer.get_batched_indexer_bwd() is aiter_marker


def test_sparse_mla_is_fixed_to_triton(monkeypatch):
    sparse_mla = importlib.import_module("lumen.ops.dsv4.sparse_mla")
    triton_marker = object()

    triton_module = types.ModuleType(
        "lumen.kernels.dsv4.sparse_mla.triton_sparse_mla"
    )
    triton_module.sparse_attn_triton = triton_marker
    monkeypatch.setitem(sys.modules, triton_module.__name__, triton_module)

    monkeypatch.setenv("V4_SPARSE_MLA_BACKEND", "tilelang")
    assert sparse_mla.get_sparse_mla_backend() == "triton"
    assert sparse_mla.get_sparse_attn_fn() is triton_marker


def test_standalone_pretrain_enables_variable_pipeline_shapes_on_config():
    source = PRETRAIN_PATH.read_text(encoding="utf-8")
    assert "config.variable_seq_lengths = True" in source
    assert "config.batch_p2p_comm = False" in source


def test_dsv4_pipeline_shape_exchange_preserves_hc_dimension():
    assert PIPELINE_PATH.is_file()
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    assert "shape_rank = 4" in source
    assert "_p2p_ops(" in source
    assert "torch.distributed.batch_isend_irecv(" not in source


def test_dsv4_ops_lazily_exports_qat_without_importing_tilekernels(monkeypatch):
    for name in tuple(sys.modules):
        if name == "tile_kernels" or name.startswith("tile_kernels."):
            monkeypatch.delitem(sys.modules, name)

    spec = importlib.util.spec_from_file_location(
        "_test_canonical_dsv4_ops_init", CANONICAL_OPS_INIT_PATH
    )
    assert spec is not None and spec.loader is not None
    canonical_ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical_ops)
    assert not any(name.startswith("tile_kernels") for name in sys.modules)

    qat_path = REPO_ROOT / "lumen/ops/dsv4/qat.py"
    spec = importlib.util.spec_from_file_location("_test_dsv4_qat", qat_path)
    assert spec is not None and spec.loader is not None
    qat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qat)
    assert callable(qat.fp8_simulate_qat)
    assert not any(name.startswith("tile_kernels") for name in sys.modules)


def test_compressor_has_no_module_level_qat_import():
    tree = ast.parse(COMPRESSOR_PATH.read_text(encoding="utf-8"))
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not isinstance(node, ast.ImportFrom)
        or node.module != "lumen.ops.dsv4.qat"
        for node in imports
    )


def test_dsv4_launchers_mount_explicit_aiter_tree():
    paths_source = (DSV4_EXAMPLES / "dsv4_paths.sh").read_text(encoding="utf-8")
    assert 'AITER_DIR="${AITER_DIR:-${WORKSPACE_ROOT}/aiter}"' in paths_source

    common = (DSV4_EXAMPLES / "dsv4_docker_common.sh").read_text(encoding="utf-8")
    assert '${AITER_DIR}:/workspace/aiter' in common
    assert "AITER_DIR=/workspace/aiter" in common

    launch = (DSV4_EXAMPLES / "dsv4_launch.sh").read_text(encoding="utf-8")
    assert "dsv4_docker_common.sh" in launch
    assert "dsv4_docker_append_kernel_env" in launch
    assert "run_dsv4_pretrain_inner.sh" in launch
    assert "run_dsv4_inner.sh" in launch

    two_node = (DSV4_EXAMPLES / "launch_dsv4_2node.sh").read_text(encoding="utf-8")
    assert 'DSV4_MODE="${DSV4_MODE:-finetune}"' in two_node
    assert "dsv4_launch.sh" in two_node
    assert '"AITER_DIR=${AITER_DIR}"' in two_node
    assert "pretrain" in two_node


def test_legacy_dsv4_wrappers_removed():
    removed = (
        "launch_dsv4_flash_pretrain_2node.sh",
        "prepare_dsv4_4layer_checkpoint.py",
        "prepare_dsv4_flash_checkpoint.py",
        "prepare_dsv4_fake_rollout.py",
    )
    for name in removed:
        assert not (DSV4_EXAMPLES / name).exists(), name


def test_pretrain_entry_installs_profiler_hook():
    source = (DSV4_EXAMPLES / "pretrain_dsv4_megatron.py").read_text(encoding="utf-8")
    assert "install_dsv4_profiler()" in source


def test_models_dsv4_package_is_canonical_entrypoint():
    tree = ast.parse(MODELS_DSV4_INIT_PATH.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "lumen.models.dsv4.compressor"
        for node in tree.body
    )
    assert "DeepSeekV4Compressor" in MODELS_DSV4_INIT_PATH.read_text(encoding="utf-8")


def test_container_preflight_requires_all_dsv4_mhc_apis():
    bootstrap = (DSV4_EXAMPLES / "bootstrap_env.sh").read_text(encoding="utf-8")
    setup = (DSV4_EXAMPLES / "setup_container_env.sh").read_text(encoding="utf-8")
    assert 'AITER_DIR="${AITER_DIR:-/workspace/aiter}"' in bootstrap
    assert "${AITER_DIR}" in bootstrap
    for symbol in ("mhc_pre_dsv4", "mhc_post_dsv4", "mhc_head_dsv4"):
        assert symbol in setup or symbol in bootstrap
    assert "missing required AIter DSV4 MHC APIs" in setup


def test_dsv4_tools_live_in_tools_package():
    tools_dir = REPO_ROOT / "lumen/tools/dsv4"
    for filename in (
        "convert_hf_to_torch_dist.py",
        "fp8_cast_bf16.py",
        "gen_fake_rollout_data.py",
        "megatron_convert.py",
        "param_name_remap.py",
        "stress_mhc_dsv4.py",
    ):
        assert (tools_dir / filename).is_file(), filename


def test_megatron_patch_is_modular():
    patch_dir = DSV4_EXAMPLES / "megatron_patch"
    assert (patch_dir / "__init__.py").is_file()
    init_source = (patch_dir / "__init__.py").read_text(encoding="utf-8")
    assert "PATCH_VERSION" in init_source
    assert "apply_all" in init_source
    patch_cli = (DSV4_EXAMPLES / "patch_rocm_megatron_dsv4.py").read_text(encoding="utf-8")
    assert "from megatron_patch import" in patch_cli


def test_unified_launcher_exists():
    launch = (DSV4_EXAMPLES / "dsv4_launch.sh").read_text(encoding="utf-8")
    assert 'DSV4_MODE="${DSV4_MODE:-finetune}"' in launch
    assert "run_dsv4_pretrain_inner.sh" in launch
    assert "run_dsv4_inner.sh" in launch
    assert (DSV4_EXAMPLES / "run_dsv4_pretrain_inner.sh").is_file()
    assert (DSV4_EXAMPLES / "run_dsv4_inner.sh").is_file()
