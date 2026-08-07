"""MHC backend selection — sync Lumen ``MHC_BACKEND`` with TileKernels dispatch.

TileKernels routes Hyper-Connection kernels via ``tile_kernels.modeling.mhc.ops``:

- ``MHC_BACKEND=triton`` (Lumen default) → ``tile_kernels/mhc/*_triton.py``
- ``MHC_BACKEND=tilelang`` → ``tile_kernels/mhc/*_kernel.py`` (TileLang)

Mount a dev tree with ``TILEKERNELS_DIR`` (see ``bootstrap_env.sh`` overlay).
"""

from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Callable

_SUPPORTED_BACKENDS = frozenset({"triton", "tilelang"})
_DEFAULT_BACKEND = "triton"
_configured = False


def get_mhc_backend() -> str:
    """Return configured MHC backend name (default: triton)."""
    backend = os.environ.get("MHC_BACKEND", _DEFAULT_BACKEND).lower()
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"MHC_BACKEND={backend!r} is unsupported; use one of {sorted(_SUPPORTED_BACKENDS)}"
        )
    return backend


def _prepend_tilekernels_path() -> None:
    root = os.environ.get("TILEKERNELS_DIR", "")
    pkg = os.path.join(root, "tile_kernels") if root else ""
    if pkg and os.path.isdir(pkg) and root not in sys.path:
        sys.path.insert(0, root)


def ensure_mhc_backend() -> str:
    """Apply ``MHC_BACKEND`` to TileKernels before importing MHC ops."""
    global _configured
    backend = get_mhc_backend()
    _prepend_tilekernels_path()
    from tile_kernels.modeling.mhc.ops.backend import get_backend, set_backend

    if not _configured:
        set_backend(backend)
        _configured = True
    active = get_backend()
    if active != backend:
        raise RuntimeError(
            f"MHC_BACKEND={backend!r} but tile_kernels loaded backend {active!r}; "
            "ensure ensure_mhc_backend() runs before importing tile_kernels.modeling.mhc.ops"
        )
    return backend


def log_mhc_backend() -> str:
    """Validate TileKernels import path and return active backend (bootstrap logging)."""
    backend = ensure_mhc_backend()
    import tile_kernels.modeling.mhc.ops  # noqa: F401

    return backend


_ops_module: ModuleType | None = None


def get_mhc_ops_module() -> ModuleType:
    """Return ``tile_kernels.modeling.mhc.ops`` after backend configuration."""
    global _ops_module
    ensure_mhc_backend()
    if _ops_module is None:
        import tile_kernels.modeling.mhc.ops as ops

        _ops_module = ops
    return _ops_module


def get_mhc_op(name: str) -> Callable:
    """Fetch a callable from ``tile_kernels.modeling.mhc.ops``."""
    op = getattr(get_mhc_ops_module(), name, None)
    if op is None:
        raise AttributeError(f"tile_kernels.modeling.mhc.ops has no attribute {name!r}")
    return op


def configure_mhc_backend(backend: str | None = None) -> str:
    """Alias for ``ensure_mhc_backend`` (patch / bootstrap_env compatibility)."""
    global _ops_module, _configured
    if backend is not None:
        os.environ["MHC_BACKEND"] = backend.lower()
        _ops_module = None
        _configured = False
    return ensure_mhc_backend()
