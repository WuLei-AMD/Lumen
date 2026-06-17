###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""P1 torch.compile compatibility tests.

Covers:
  - try_backends compile-mode fast path: skips try/except, calls backends[0]
  - try_backends compile-mode uses cached backend when available
  - try_backends eager path unaffected (regression)
  - _mark_allow_in_graph: _LayerNormGradQuant / _RMSNormGradQuant registered
  - fullgraph=True does not break on LayerNorm / RMSNorm grad-quant path (CUDA)
"""

import sys
from unittest.mock import patch

import pytest
import torch

from lumen.ops.dispatch import Backend, _backend_cache, _mark_allow_in_graph, try_backends

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain(*results_or_exc):
    """Build a fake backend chain from return values or exception instances."""
    backends = [Backend.ASM, Backend.CK, Backend.TRITON]
    chain = []
    for backend, val in zip(backends, results_or_exc):
        if isinstance(val, BaseException):
            exc = val

            def fn(e=exc):
                raise e

        else:

            def fn(v=val):
                return v

        chain.append((backend, fn))
    return chain


def _clear_cache(op_name):
    for key in [op_name, op_name + ":hits", op_name + ":prev"]:
        _backend_cache.pop(key, None)


# ---------------------------------------------------------------------------
# try_backends — compile-mode fast path
# ---------------------------------------------------------------------------


class TestTryBackendsCompileMode:
    def setup_method(self):
        _clear_cache("compile_test")

    def test_calls_backends_0_when_no_cache(self):
        """In compile mode with no cached backend, backends[0] is called directly."""
        called = []
        chain = [
            (Backend.ASM, lambda: called.append("asm") or "asm_result"),
            (Backend.CK, lambda: called.append("ck") or "ck_result"),
        ]
        with patch("torch.compiler.is_compiling", return_value=True):
            result = try_backends(chain, op_name="compile_test")

        assert result == "asm_result"
        assert called == ["asm"], "only backends[0] must be called in compile mode"

    def test_skips_fallback_even_when_backends_0_would_fail_in_eager(self):
        """Compile mode calls backends[0] unconditionally — no exception-driven fallback."""
        fail_count = []

        def _fail():
            fail_count.append(1)
            # In compile mode this exception is NOT caught — it propagates up.
            raise RuntimeError("simulated ASM failure")

        chain = [
            (Backend.ASM, _fail),
            (Backend.CK, lambda: "ck_result"),
        ]
        with patch("torch.compiler.is_compiling", return_value=True):
            with pytest.raises(RuntimeError, match="simulated ASM failure"):
                try_backends(chain, op_name="compile_test")

        assert len(fail_count) == 1

    def test_uses_cached_backend_in_compile_mode(self):
        """If warmup already locked a backend, compile mode uses the cached index."""
        called = []
        chain = [
            (Backend.ASM, lambda: called.append("asm") or "asm"),
            (Backend.CK, lambda: called.append("ck") or "ck"),
            (Backend.TRITON, lambda: called.append("tri") or "tri"),
        ]
        # Simulate warmup having locked CK (index=1).
        _backend_cache["compile_test"] = 1

        with patch("torch.compiler.is_compiling", return_value=True):
            result = try_backends(chain, op_name="compile_test")

        assert result == "ck"
        assert called == ["ck"], "must use cached index, not backends[0]"

    def test_falls_back_to_0_when_cached_index_out_of_range(self):
        """Stale cache index beyond chain length falls back to backends[0]."""
        called = []
        chain = [
            (Backend.CK, lambda: called.append("ck") or "ck"),
        ]
        _backend_cache["compile_test"] = 5  # out of range

        with patch("torch.compiler.is_compiling", return_value=True):
            result = try_backends(chain, op_name="compile_test")

        assert result == "ck"
        assert called == ["ck"]


# ---------------------------------------------------------------------------
# try_backends — eager path regression
# ---------------------------------------------------------------------------


class TestTryBackendsEagerRegression:
    def setup_method(self):
        _clear_cache("eager_reg")

    def test_eager_still_falls_through_on_error(self):
        """Eager path is unchanged: falls through RuntimeError to next backend."""
        chain = _make_chain(RuntimeError("asm fail"), "ck_ok")
        result = try_backends(chain, op_name="eager_reg")
        assert result == "ck_ok"

    def test_eager_returns_first_success(self):
        chain = _make_chain("asm_ok", "ck_ok")
        result = try_backends(chain, op_name="eager_reg")
        assert result == "asm_ok"

    def test_eager_raises_when_all_fail(self):
        chain = _make_chain(RuntimeError("a"), RuntimeError("b"), RuntimeError("c"))
        with pytest.raises(RuntimeError, match="all AITER backends exhausted"):
            try_backends(chain, op_name="eager_reg")


# ---------------------------------------------------------------------------
# _mark_allow_in_graph — registration check
# ---------------------------------------------------------------------------


class TestMarkAllowInGraph:
    def test_registers_without_error(self):
        """_mark_allow_in_graph does not raise on any registered Function."""
        import torch.autograd

        class _Dummy(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, g):
                return g

        _mark_allow_in_graph(_Dummy)  # must not raise

    def test_layernorm_gradquant_is_registered(self):
        """_LayerNormGradQuant must be in Dynamo's allow-list."""
        pytest.importorskip("torch._dynamo")
        from lumen.ops.normalization.layernorm import _LayerNormGradQuant

        # allow_in_graph stores the class in _allowed_callable_ids.
        # The simplest proxy: compiling a wrapper that calls .apply()
        # with fullgraph=True must not raise InternalTorchDynamoError.
        # We verify registration indirectly via the Dynamo lookup table.
        try:
            from torch._dynamo.trace_rules import is_callable_allowed
            assert is_callable_allowed(_LayerNormGradQuant), (
                "_LayerNormGradQuant not in Dynamo allow-list; "
                "_mark_allow_in_graph was not called"
            )
        except ImportError:
            # Older PyTorch without trace_rules — skip introspection check.
            pytest.skip("torch._dynamo.trace_rules not available")

    def test_rmsnorm_gradquant_is_registered(self):
        """_RMSNormGradQuant must be in Dynamo's allow-list."""
        pytest.importorskip("torch._dynamo")
        from lumen.ops.normalization.rmsnorm import _RMSNormGradQuant

        try:
            from torch._dynamo.trace_rules import is_callable_allowed
            assert is_callable_allowed(_RMSNormGradQuant), (
                "_RMSNormGradQuant not in Dynamo allow-list; "
                "_mark_allow_in_graph was not called"
            )
        except ImportError:
            pytest.skip("torch._dynamo.trace_rules not available")


# ---------------------------------------------------------------------------
# fullgraph=True integration — requires CUDA
# ---------------------------------------------------------------------------


@_CUDA
class TestFullgraphNorm:
    """Verify no graph break under fullgraph=True for grad-quant norm paths."""

    @staticmethod
    def _compile_counter():
        """Return a simple compile-frame counter backend."""
        counts = {"frames": 0}

        def _backend(gm, example_inputs):
            counts["frames"] += 1
            return gm.forward

        _backend.counts = counts
        return _backend

    def test_layernorm_gradquant_fullgraph(self):
        """LayerNorm grad-quant path compiles without graph breaks."""
        from lumen.ops.normalization.layernorm import layernorm

        hidden = 256
        x = torch.randn(4, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.ones(hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        b = torch.zeros(hidden, device="cuda", dtype=torch.bfloat16)

        def _fn(x, w, b):
            return layernorm(x, w, b, eps=1e-5, grad_quant_type="bf16")

        backend = self._compile_counter()
        compiled = torch.compile(_fn, backend=backend, fullgraph=True)
        # Warm up eager first to lock the backend cache.
        for _ in range(3):
            _fn(x, w, b)

        out = compiled(x, w, b)
        assert out.shape == x.shape
        # One frame = no graph breaks.
        assert backend.counts["frames"] == 1, (
            f"Expected 1 compiled frame, got {backend.counts['frames']} — "
            "graph break detected in layernorm grad-quant path"
        )

    def test_rmsnorm_gradquant_fullgraph(self):
        """RMSNorm grad-quant path compiles without graph breaks."""
        from lumen.ops.normalization.rmsnorm import rmsnorm

        hidden = 256
        x = torch.randn(4, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.ones(hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def _fn(x, w):
            return rmsnorm(x, w, eps=1e-6, grad_quant_type="bf16")

        backend = self._compile_counter()
        compiled = torch.compile(_fn, backend=backend, fullgraph=True)
        for _ in range(3):
            _fn(x, w)

        out = compiled(x, w)
        assert out.shape == x.shape
        assert backend.counts["frames"] == 1, (
            f"Expected 1 compiled frame, got {backend.counts['frames']} — "
            "graph break detected in rmsnorm grad-quant path"
        )

    def test_try_backends_compile_mode_output_matches_eager(self):
        """Compile-mode fast path produces the same result as eager."""
        results = []
        chain = [
            (Backend.CK, lambda: torch.tensor(42.0)),
        ]
        _clear_cache("match_test")

        # Eager
        eager_result = try_backends(chain, op_name="match_test")
        results.append(eager_result.item())

        # Compile mode (simulated)
        _clear_cache("match_test")
        with patch("torch.compiler.is_compiling", return_value=True):
            compile_result = try_backends(chain, op_name="match_test")
        results.append(compile_result.item())

        assert results[0] == results[1]
