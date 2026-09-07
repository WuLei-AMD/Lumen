###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""HIP/CUDA graph capture utilities for Lumen.

Provides graph-capture wrappers that record a training step (or sub-step)
into a replayable HIP graph, reducing kernel launch overhead. Handles
FP8 scaling manager state updates that must remain graph-safe.

Key classes:

- ``LumenGraphedCallable`` — wraps a single callable in a forward-only graph.
- ``LumenGraphedModule``   — wraps an ``nn.Module`` via ``LumenGraphedCallable``.
- ``LumenGraphedLayer``    — per-layer forward+backward graph capture with a
  custom ``autograd.Function`` that replays separate forward and backward
  CUDA graphs.  Modelled after TE's ``Graphed`` autograd function.
- ``capture_lumen_graphs`` — captures all transformer layers of a Megatron
  GPTModel and replaces their ``forward`` with graphed wrappers.
"""

import functools
import logging
import math
import os
import time
import weakref
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_PENDING_ATTENTION_GRAPHS = weakref.WeakSet()
_ALL_ATTENTION_GRAPH_RUNNERS = weakref.WeakSet()
_ATTENTION_GRAPH_TOTAL_LAYERS = 0
_LAYER_STATIC_HIDDEN: Dict[Tuple[int, int], torch.Tensor] = {}
# RoPE / mask tensors may be shared across layers of one microbatch.
_MB_STATIC_KWARGS: Dict[int, Dict[str, Any]] = {}
_MB_KWARG_SIGNATURES: Dict[int, Dict[str, Tuple[Any, ...]]] = {}
_MB_CONSTANT_KWARGS: Dict[int, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Forward-only graph capture (unchanged from original)
# ---------------------------------------------------------------------------


class LumenGraphedCallable:
    """Wraps a callable (forward pass) in a CUDA/HIP graph.

    1. Warm up the callable with sample inputs
    2. Record a graph capturing the computation
    3. Replay the graph on subsequent calls

    The graph is re-recorded if input shapes change.

    Args:
        callable_fn: The function to graph-capture.
        sample_args: Sample arguments for warmup and capture.
        sample_kwargs: Sample keyword arguments.
        num_warmup: Number of warmup iterations before capture.
        pool: Optional CUDA memory pool for graph allocation.
    """

    def __init__(
        self,
        callable_fn: Callable,
        sample_args: Tuple[torch.Tensor, ...],
        sample_kwargs: Optional[dict] = None,
        num_warmup: int = 3,
        pool: Optional[Any] = None,
    ):
        self._fn = callable_fn
        self._num_warmup = num_warmup
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._static_output: Optional[torch.Tensor] = None
        self._input_shapes: Optional[Tuple[torch.Size, ...]] = None
        self._pool = pool

        self._capture(sample_args, sample_kwargs or {})

    def _capture(self, args: Tuple, kwargs: dict):
        """Warm up and capture the graph."""
        device = None
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_cuda:
                device = a.device
                break

        if device is None:
            logger.warning("No CUDA tensors in args, skipping graph capture")
            self._graph = None
            return

        self._static_inputs = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
        self._input_shapes = tuple(a.shape if isinstance(a, torch.Tensor) else None for a in args)

        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(self._num_warmup):
                self._fn(*self._static_inputs, **kwargs)
        torch.cuda.current_stream(device).wait_stream(s)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=self._pool, stream=s):
            self._static_output = self._fn(*self._static_inputs, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        if self._graph is None:
            return self._fn(*args, **kwargs)

        for static, new in zip(self._static_inputs, args):
            if isinstance(static, torch.Tensor) and isinstance(new, torch.Tensor):
                if static.shape != new.shape:
                    logger.info("Input shape changed, re-capturing graph")
                    self._capture(args, kwargs)
                    return self._static_output
                static.copy_(new)

        self._graph.replay()
        return self._static_output

    def reset(self):
        """Release the captured graph."""
        self._graph = None
        self._static_inputs = None
        self._static_output = None


class LumenGraphedModule(nn.Module):
    """Wrapper that graph-captures a module's forward pass.

    Args:
        module: The module to wrap.
        sample_input: Sample input for graph capture.
        num_warmup: Warmup iterations.
        enabled: Whether graph capture is active.
    """

    def __init__(
        self,
        module: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        num_warmup: int = 3,
        enabled: bool = True,
    ):
        super().__init__()
        self.module = module
        self.enabled = enabled
        self._graphed: Optional[LumenGraphedCallable] = None
        self._num_warmup = num_warmup

        if enabled and sample_input is not None:
            self._graphed = LumenGraphedCallable(module, (sample_input,), num_warmup=num_warmup)

    def forward(self, *args, **kwargs):
        if self._graphed is not None and self.enabled:
            return self._graphed(*args, **kwargs)
        return self.module(*args, **kwargs)

    def capture(self, sample_input: torch.Tensor):
        """Manually trigger graph capture with the given sample input."""
        self._graphed = LumenGraphedCallable(self.module, (sample_input,), num_warmup=self._num_warmup)

    def release_graph(self):
        """Release the captured graph."""
        if self._graphed is not None:
            self._graphed.reset()
            self._graphed = None


def lumen_make_graphed_callables(
    callables: List[Callable],
    sample_args: List[Tuple[torch.Tensor, ...]],
    num_warmup: int = 3,
) -> List[LumenGraphedCallable]:
    """Graph-capture multiple callables sharing a memory pool.

    Args:
        callables: List of functions to capture.
        sample_args: Corresponding sample arguments.
        num_warmup: Warmup iterations per callable.

    Returns:
        List of LumenGraphedCallable instances.
    """
    pool = getattr(torch.cuda, "graph_pool_handle", lambda: None)() if torch.cuda.is_available() else None
    graphed = []
    for fn, args in zip(callables, sample_args):
        graphed_callable = LumenGraphedCallable(fn, args, num_warmup=num_warmup, pool=pool)
        graphed.append(graphed_callable)
    return graphed


# ---------------------------------------------------------------------------
# Per-layer forward+backward graph capture
# ---------------------------------------------------------------------------


class _FwdGraphedLayerFn(torch.autograd.Function):
    """Forward-only graph replay with eager backward.

    Forward: replays the captured CUDA graph (fast, reduced kernel-launch
    overhead).  Backward: re-runs the layer forward eagerly to build a fresh
    autograd tape, then calls ``torch.autograd.backward`` on that tape.  This
    avoids capturing the backward graph (which requires a full extra
    forward+backward warmup at 98 %+ memory utilization) while still getting
    the forward kernel-launch savings.
    """

    @staticmethod
    def forward(
        ctx,
        real_input,
        static_fwd_input,
        static_fwd_output,
        fwd_graph,
        layer_fwd_fn,
        static_kwargs,
    ):
        ctx.layer_fwd_fn = layer_fwd_fn
        ctx.static_kwargs = static_kwargs
        ctx.save_for_backward(real_input)

        static_fwd_input.copy_(real_input)
        fwd_graph.replay()

        return static_fwd_output.detach().clone()

    @staticmethod
    def backward(ctx, grad_output):
        (real_input,) = ctx.saved_tensors
        inp = real_input.detach().requires_grad_(True)

        with torch.enable_grad():
            out = ctx.layer_fwd_fn(inp, **ctx.static_kwargs)
            if isinstance(out, tuple):
                out = out[0]
            torch.autograd.backward(out, grad_output)

        return (inp.grad, None, None, None, None, None)


class LumenGraphedLayer:
    """Per-layer forward+backward graph capture with lazy initialization.

    Graph capture is deferred until the first real training call so that
    warmup happens on the *actual* data path (priming Triton/AITER JIT
    caches) and capture occurs only after the memory pool is established.

    Uses ``torch.autograd.grad(only_inputs=True)`` for warmup and backward
    capture to avoid "Cannot set grad twice" errors with
    ``gradient_accumulation_fusion`` and FP8 parameter storage.
    Saves and restores ``main_grad`` across warmup.

    Modelled after Megatron-LM's ``_CudaGraphRunner``.

    Args:
        layer: The ``nn.Module`` transformer layer.
        num_warmup: Eager forward+backward passes before graph capture.
    """

    _shared_pool = None

    @classmethod
    def get_shared_pool(cls):
        """Return a shared graph pool handle for all graphed layers.

        Sharing a pool allows ROCm to reuse memory between layers since
        they execute sequentially — one layer's intermediates can overlap
        with another's in the pool.
        """
        if cls._shared_pool is None and torch.cuda.is_available():
            cls._shared_pool = getattr(torch.cuda, "graph_pool_handle", lambda: None)()
        return cls._shared_pool

    def __init__(self, layer: nn.Module, num_warmup: int = 3):
        self.layer = layer
        self._original_forward = layer.forward
        self._num_warmup = num_warmup
        self._call_count = 0
        self._captured = False

        self._fwd_graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None
        self._static_kwargs: Optional[Dict[str, Any]] = None
        self._extra_outputs: tuple = ()

        self._pool = self.get_shared_pool()

    def _do_capture(
        self,
        hidden_states: torch.Tensor,
        kwargs: Dict[str, Any],
    ) -> Any:
        """Capture forward-only graph. No warmup (layer already warmed by
        num_warmup real training steps). Backward runs eagerly via recompute."""
        device = hidden_states.device
        fwd = self._original_forward

        self._static_input = hidden_states.clone().detach()
        self._static_kwargs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}

        from lumen.ops.quantize.linear import set_graph_capture_mode

        set_graph_capture_mode(True)

        try:
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream(device))

            self._fwd_graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(device)
            with torch.no_grad(), torch.cuda.graph(
                self._fwd_graph,
                pool=self._pool,
                stream=s,
                capture_error_mode="thread_local",
            ):
                fwd_out = fwd(self._static_input, **self._static_kwargs)
            torch.cuda.current_stream(device).wait_stream(s)
        finally:
            set_graph_capture_mode(False)

        if isinstance(fwd_out, tuple):
            self._static_output = fwd_out[0]
            self._extra_outputs = fwd_out[1:]
        else:
            self._static_output = fwd_out
            self._extra_outputs = ()

        self._captured = True
        return self._replay(hidden_states, kwargs)

    def _replay(
        self,
        hidden_states: torch.Tensor,
        kwargs: Dict[str, Any],
    ) -> Any:
        """Replay captured forward graph; backward is eager via recompute."""
        for k, v in kwargs.items():
            if k in self._static_kwargs and isinstance(v, torch.Tensor):
                self._static_kwargs[k].copy_(v)

        out = _FwdGraphedLayerFn.apply(
            hidden_states,
            self._static_input,
            self._static_output,
            self._fwd_graph,
            self._original_forward,
            self._static_kwargs,
        )
        if self._extra_outputs:
            return (out,) + self._extra_outputs
        return out

    def __call__(self, hidden_states, **kwargs):
        if self._captured:
            return self._replay(hidden_states, kwargs)

        if not torch.is_grad_enabled():
            return self._original_forward(hidden_states, **kwargs)

        self._call_count += 1

        if self._call_count <= self._num_warmup:
            return self._original_forward(hidden_states, **kwargs)

        from lumen.ops.dispatch import _backend_cache

        uncached = [k for k in _backend_cache if k.endswith(":prev") and k[:-5] not in _backend_cache]
        if uncached:
            logger.debug("Deferring capture: %d ops not yet cached", len(uncached))
            return self._original_forward(hidden_states, **kwargs)

        try:
            result = self._do_capture(hidden_states, kwargs)
            logger.info(
                "Graph capture succeeded for layer after %d calls",
                self._call_count,
            )
            return result
        except Exception as e:
            logger.warning("Graph capture failed: %s — permanent eager fallback", e)
            self._captured = False
            if self._fwd_graph is not None:
                del self._fwd_graph
                self._fwd_graph = None
            self._static_input = None
            self._static_output = None
            self._static_kwargs = None
            torch.cuda.empty_cache()
            self._num_warmup = float("inf")
            return self._original_forward(hidden_states, **kwargs)

    def disable(self):
        self._captured = False
        self._fwd_graph = None
        self._call_count = 0

    def enable(self):
        pass


def install_lazy_graph_capture(
    model: nn.Module,
    num_warmup: int = 3,
    skip_recomputed_layers: int = 0,
    max_graphed_layers: int = 0,
) -> int:
    """Replace non-checkpointed transformer layers' forward with a lazy
    graph-capture wrapper.

    Layers using activation checkpointing (recompute) are **skipped**
    because the checkpoint backward re-runs the forward, which is
    incompatible with graph replay's inplace static-buffer updates.

    Args:
        model: Megatron GPTModel with ``.decoder.layers``.
        num_warmup: Number of eager forward passes before capture.
        skip_recomputed_layers: Number of leading layers to skip
            (matches ``recompute_num_layers`` from Megatron config).
        max_graphed_layers: Maximum number of layers to graph (0 = all eligible).
            Limits memory usage in the graph pool at high memory utilization.

    Returns:
        Number of layers wrapped.
    """
    if not hasattr(model, "decoder") or model.decoder is None:
        logger.warning("Model has no decoder attribute, skipping graph wrappers")
        return 0
    if not hasattr(model.decoder, "layers"):
        logger.warning("Model decoder has no layers, skipping graph wrappers")
        return 0

    layers = model.decoder.layers
    wrapped = 0
    for l_no, layer in enumerate(layers):
        if l_no < skip_recomputed_layers:
            continue
        if max_graphed_layers > 0 and wrapped >= max_graphed_layers:
            break
        graphed = LumenGraphedLayer(layer, num_warmup=num_warmup)
        layer.forward = graphed
        wrapped += 1

    logger.info(
        f"Installed lazy graph capture on {wrapped}/{len(layers)} transformer layers "
        f"(skipped {skip_recomputed_layers} recomputed, "
        f"max {max_graphed_layers if max_graphed_layers > 0 else 'unlimited'}, "
        f"capture after {num_warmup} warmup steps)"
    )
    return wrapped


# ---------------------------------------------------------------------------
# Attention-only forward+backward graph capture
# ---------------------------------------------------------------------------


def _tensor_signature(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Properties that must stay fixed across graph replays."""
    return (tensor.shape, tensor.dtype, tensor.device, tensor.layout, tensor.stride())


def _is_retryable_attention_graph_error(error: BaseException) -> bool:
    text = str(error).lower()
    return (
        "unjoined" in text
        or "invalid argument" in text
        or "invalidvalue" in text
        or "hiperrorinvalidvalue" in text
    )


def _clear_async_hip_error() -> None:
    """Drop a sticky HIP last-error so a failed capture cannot poison RCCL."""
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    try:
        torch.cuda.cudart().cudaGetLastError()
    except Exception:
        pass


def _synchronize_for_attention_graph_capture(device=None) -> None:
    """Join every stream on this device before HIP graph capture."""
    if device is None:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize(device)


_CAPTURES_PER_STEP = None
_SHARE_ROPE = None
_CHECK_NUMERICS = os.environ.get("LUMEN_ATTN_GRAPH_CHECK_NUMERICS", "0") == "1"
_PROFILE_REPLAY = os.environ.get("LUMEN_ATTN_GRAPH_PROFILE", "0") == "1"
_REPLAY_STATS = {
    "fwd_copy_us": 0.0,
    "fwd_replay_us": 0.0,
    "bwd_copy_us": 0.0,
    "bwd_replay_us": 0.0,
    "count": 0,
}
_HOST_REPLAY_STATS = {"fwd_ns": 0, "bwd_ns": 0, "count": 0}


def _record_host_replay(kind: str, started_ns: int) -> None:
    _HOST_REPLAY_STATS[kind] += time.perf_counter_ns() - started_ns
    if kind == "bwd_ns":
        _HOST_REPLAY_STATS["count"] += 1
        n = _HOST_REPLAY_STATS["count"]
        if n % 768 == 0:
            logger.info(
                "Lumen attention graph host replay stats over %d pairs: "
                "fwd %.1f us | bwd %.1f us",
                n,
                _HOST_REPLAY_STATS["fwd_ns"] / n / 1000.0,
                _HOST_REPLAY_STATS["bwd_ns"] / n / 1000.0,
            )


def _captures_per_step() -> int:
    global _CAPTURES_PER_STEP
    if _CAPTURES_PER_STEP is None:
        value = int(os.environ.get("LUMEN_ATTN_GRAPH_CAPTURES_PER_STEP", "0"))
        _CAPTURES_PER_STEP = value if value > 0 else 10**9
    return _CAPTURES_PER_STEP


def _share_rope() -> bool:
    global _SHARE_ROPE
    if _SHARE_ROPE is None:
        _SHARE_ROPE = os.environ.get("LUMEN_ATTN_GRAPH_SHARE_ROPE", "1") != "0"
    return _SHARE_ROPE


def _record_replay_pair(copy_start, copy_end, replay_end, copy_key, replay_key) -> None:
    replay_end.synchronize()
    _REPLAY_STATS[copy_key] += copy_start.elapsed_time(copy_end) * 1000.0
    _REPLAY_STATS[replay_key] += copy_end.elapsed_time(replay_end) * 1000.0
    if replay_key == "bwd_replay_us":
        _REPLAY_STATS["count"] += 1
        n = _REPLAY_STATS["count"]
        if n % 768 == 0:
            logger.info(
                "Lumen attention graph GPU replay stats over %d pairs: "
                "fwd copy %.1f us replay %.1f us | bwd copy %.1f us replay %.1f us",
                n,
                _REPLAY_STATS["fwd_copy_us"] / n,
                _REPLAY_STATS["fwd_replay_us"] / n,
                _REPLAY_STATS["bwd_copy_us"] / n,
                _REPLAY_STATS["bwd_replay_us"] / n,
            )


def _num_microbatches() -> int:
    try:
        from megatron.core.num_microbatches_calculator import get_num_microbatches

        return max(int(get_num_microbatches()), 0)
    except Exception:
        return 0


def _layer_in_snapshot_cohort(layer_number: int) -> bool:
    """Snapshot whole layers, never a 1F1B prefix of every layer.

    ``captures_per_step=96`` with 48 layers used to snapshot mb0+mb1 of
    *all* layers (96 calls), then replay a mix of graphed and eager
    microbatches on the same layer — that is the GBS=256 NaN.
    """
    n_mb = _num_microbatches()
    cap = _captures_per_step()
    max_layers = max(cap // n_mb, 1) if n_mb > 0 else 10**9
    by_layer: Dict[int, List[Any]] = {}
    for runner in list(_ALL_ATTENTION_GRAPH_RUNNERS):
        if getattr(runner, "_disabled", False):
            continue
        ln = int(getattr(runner, "_layer_number", 10**9))
        by_layer.setdefault(ln, []).append(runner)
    uncaptured = []
    for ln in sorted(by_layer):
        runners = by_layer[ln]
        if not all(getattr(r, "_captured", False) for r in runners):
            uncaptured.append(ln)
    return int(layer_number) in uncaptured[:max_layers]


def _pending_snapshot_count() -> int:
    return sum(
        1
        for runner in list(_PENDING_ATTENTION_GRAPHS)
        if getattr(runner, "_capture_input", None) is not None
    )


def _copy_into_static(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Fill a CUDA-graph leaf without an autograd in-place on ``requires_grad``.

    ``Tensor.copy_`` on a leaf that requires grad raises
    ``a leaf Variable that requires grad is being used in an in-place
    operation``. Graph replay/capture still need to overwrite the same
    storage, so write through ``.data``.
    """
    if dst.data_ptr() != src.data_ptr():
        dst.data.copy_(src)


def _schedule_reused_hidden(
    layer_number: int,
    microbatch: int,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Return one of two live input slots for an EP-overlapped layer.

    In the combined PP=1 schedule, mb(N+2) forward starts only after mb(N)
    backward has consumed this layer's input. TE's input reuse analysis reaches
    the same two-slot result.
    """
    key = (int(layer_number), int(microbatch) % 2)
    static = _LAYER_STATIC_HIDDEN.get(key)
    if static is None:
        static = hidden_states.detach().clone()
        static.requires_grad_(True)
        _LAYER_STATIC_HIDDEN[key] = static
    else:
        if _tensor_signature(static) != _tensor_signature(hidden_states):
            raise RuntimeError(
                "Lumen attention graph hidden-state signature changed across microbatches"
            )
        _copy_into_static(static, hidden_states)
    return static


_TORCH_DTYPE_TO_ARRAY_TYPESTR = {
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.int64: "<i8",
    torch.int32: "<i4",
    torch.int8: "|i1",
    torch.bool: "|b1",
    torch.bfloat16: "<f2",
}


class _WeakRefTensor:
    """CUDA-array-interface view that does not own graph-pool storage."""

    def __init__(self, tensor: torch.Tensor):
        self._data_ptr = tensor.data_ptr()
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)

    def data_ptr(self):
        return self._data_ptr

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": self.shape,
            "typestr": _TORCH_DTYPE_TO_ARRAY_TYPESTR[self.dtype],
            "data": (self._data_ptr if math.prod(self.shape) > 0 else 0, False),
            "version": 3,
        }


def _make_weak_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Release allocator ownership while preserving a replayable tensor view."""
    if tensor is None or not tensor.is_cuda:
        return tensor
    old_ptr = tensor.data_ptr()
    view = torch.as_tensor(_WeakRefTensor(tensor)).view(tensor.dtype)
    if view.data_ptr() != old_ptr:
        raise RuntimeError("Graph weak tensor changed its data pointer")
    return view


def _shared_static_kwargs(microbatch: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """RoPE / mask tensors are read-only and identical across layers."""
    cached = _MB_STATIC_KWARGS.get(microbatch)
    if cached is None:
        cached, signatures, constants = {}, {}, {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                cached[key] = value.detach().clone()
                signatures[key] = _tensor_signature(value)
            else:
                cached[key] = value
                constants[key] = value
        _MB_STATIC_KWARGS[microbatch] = cached
        _MB_KWARG_SIGNATURES[microbatch] = signatures
        _MB_CONSTANT_KWARGS[microbatch] = constants
        return cached
    signatures = _MB_KWARG_SIGNATURES[microbatch]
    for key, value in kwargs.items():
        if key in signatures and isinstance(value, torch.Tensor):
            _copy_into_static(cached[key], value)
        elif key in _MB_CONSTANT_KWARGS[microbatch]:
            cached[key] = value
    return cached


class _AttentionGraphReplay(torch.autograd.Function):
    """Connect a pair of forward/backward HIP graphs to PyTorch autograd.

    Parameters are Function inputs, matching TE's ``make_graphed_callables``.
    Returning their static grads lets PyTorch's AccumulateGrad nodes and
    Megatron DDP hooks run normally after graph replay.
    """

    @staticmethod
    def forward(ctx, runner, hidden_states, *parameters):
        del parameters
        ctx.runner = runner
        static_input = runner._static_input
        started_ns = time.perf_counter_ns()
        if _PROFILE_REPLAY:
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            replay_end = torch.cuda.Event(enable_timing=True)
            copy_start.record()
            if static_input.data_ptr() != hidden_states.data_ptr():
                static_input.data.copy_(hidden_states)
            copy_end.record()
            runner._fwd_graph.replay()
            replay_end.record()
            _record_replay_pair(
                copy_start, copy_end, replay_end, "fwd_copy_us", "fwd_replay_us"
            )
        else:
            if static_input.data_ptr() != hidden_states.data_ptr():
                static_input.data.copy_(hidden_states)
            runner._fwd_graph.replay()
        _record_host_replay("fwd_ns", started_ns)
        # Megatron ``_CudagraphReplayNode`` returns the static surface directly.
        # ``.detach()`` on ROCm triggers ``hipEventSynchronize`` (~12 ms × 768)
        # and drains the GPU pipeline, killing EP / param-gather overlap.
        return runner._static_output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        runner = ctx.runner
        static_grad_output = runner._static_grad_output
        started_ns = time.perf_counter_ns()
        if _PROFILE_REPLAY:
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            replay_end = torch.cuda.Event(enable_timing=True)
            copy_start.record()
            if static_grad_output.data_ptr() != grad_output.data_ptr():
                static_grad_output.copy_(grad_output)
            copy_end.record()
            runner._bwd_graph.replay()
            replay_end.record()
            _record_replay_pair(
                copy_start, copy_end, replay_end, "bwd_copy_us", "bwd_replay_us"
            )
        else:
            if static_grad_output.data_ptr() != grad_output.data_ptr():
                static_grad_output.copy_(grad_output)
            runner._bwd_graph.replay()
        _record_host_replay("bwd_ns", started_ns)
        if _CHECK_NUMERICS:
            torch.cuda.synchronize()
            tensors = (
                ("grad_output", runner._static_grad_output),
                ("grad_input", runner._static_grad_input),
                *(
                    (f"parameter_grad[{index}]", grad)
                    for index, grad in enumerate(runner._static_parameter_grads)
                ),
            )
            for label, tensor in tensors:
                if tensor is not None and not torch.isfinite(tensor).all():
                    logger.error("%s replay produced non-finite %s", runner._name, label)
        grad_input = runner._static_grad_input
        # The autograd engine consumes parameter grads in AccumulateGrad before
        # returning from this backward invocation, so they do not need a
        # per-replay clone. This matches TE's Graphed.backward lifetime
        # protocol. The capture-order weak refs only alias buffers whose
        # backward invocations do not overlap.
        return (
            None,
            grad_input,
            *runner._static_parameter_grads,
        )


class LumenGraphedAttention:
    """Lazily capture a Megatron layer's ``_forward_attention`` fwd/bwd.

    This is intentionally narrower than :class:`LumenGraphedLayer`: the MoE/MLP
    half of the layer remains eager, so dynamic expert-token shapes and EP
    all-to-all communication never enter the graph.

    One runner is installed per ``(layer, microbatch)``. Memory matches TE more
    closely than the first Lumen skeleton:

    * parameters are autograd Function inputs, so DDP hooks run naturally;
    * at most ``LUMEN_ATTN_GRAPH_CAPTURES_PER_STEP`` input snapshots are live;
    * snapshots fill whole layers (all microbatches) before starting the next;
    * RoPE is per-runner unless ``LUMEN_ATTN_GRAPH_SHARE_ROPE=1``;
    * static hidden / dY stay per ``(layer, microbatch)``.

    Current scope limitations are explicit:

    * one tensor output plus an optional ``None`` context;
    * tensor arguments must retain shape, dtype, device, layout, and stride;
    * non-tensor arguments must remain unchanged after capture;
    * parameter mutation between capture and replay is supported, but replacing
      a Parameter object is not.
    """

    _shared_pool = None
    @classmethod
    def get_shared_pool(cls):
        """One pool, captured in schedule order like TE."""
        if cls._shared_pool is None and torch.cuda.is_available():
            cls._shared_pool = getattr(torch.cuda, "graph_pool_handle", lambda: None)()
        return cls._shared_pool

    def __init__(
        self,
        forward_attention: Callable,
        parameters: Iterable[torch.nn.Parameter],
        *,
        num_warmup: int = 3,
        name: str = "attention",
        microbatch: int = 0,
        layer_number: int = 0,
    ):
        self._forward_attention = forward_attention
        self._parameters = tuple(p for p in parameters if p.requires_grad)
        self._num_warmup = max(int(num_warmup), 1)
        self._name = name
        self._layer_number = int(layer_number)
        self._microbatch = int(microbatch)
        _ALL_ATTENTION_GRAPH_RUNNERS.add(self)
        self._call_count = 0
        self._captured = False
        self._disabled = False
        self._capture_ready = False
        self._capture_input: Optional[torch.Tensor] = None
        self._capture_kwargs: Dict[str, Any] = {}

        self._fwd_graph: Optional[torch.cuda.CUDAGraph] = None
        self._bwd_graph: Optional[torch.cuda.CUDAGraph] = None
        self._pool = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_kwargs: Dict[str, Any] = {}
        self._static_output: Optional[torch.Tensor] = None
        self._static_grad_output: Optional[torch.Tensor] = None
        self._static_grad_input: Optional[torch.Tensor] = None
        self._static_parameter_grads: Tuple[Optional[torch.Tensor], ...] = ()
        self._input_signature: Optional[Tuple[Any, ...]] = None
        self._kwarg_signatures: Dict[str, Tuple[Any, ...]] = {}
        self._constant_kwargs: Dict[str, Any] = {}

    @property
    def captured(self) -> bool:
        return self._captured

    def _split_inputs(self, args, kwargs):
        kwargs = kwargs.copy()
        if args:
            hidden_states = args[0]
            trailing_args = args[1:]
            if trailing_args:
                raise RuntimeError(
                    "Lumen attention graphs require non-hidden-state inputs as keyword arguments"
                )
        else:
            hidden_states = kwargs.pop("hidden_states")
        if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda:
            raise RuntimeError("Lumen attention graphs require a CUDA hidden_states tensor")
        return hidden_states, kwargs

    def _clone_kwargs(self, kwargs):
        static_kwargs = {}
        self._kwarg_signatures = {}
        self._constant_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                static_kwargs[key] = value.detach().clone()
                self._kwarg_signatures[key] = _tensor_signature(value)
            elif value is None or isinstance(value, (bool, int, float, str)):
                static_kwargs[key] = value
                self._constant_kwargs[key] = value
            else:
                raise RuntimeError(
                    f"Lumen attention graphs do not support kwarg {key!r} of type "
                    f"{type(value).__name__}"
                )
        return static_kwargs

    @staticmethod
    def _primary_output(output):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple) and len(output) == 2:
            hidden_states, context = output
            if context is not None:
                raise RuntimeError("Lumen attention graphs do not yet support cross-attention context")
            if isinstance(hidden_states, torch.Tensor):
                return hidden_states
        raise RuntimeError(
            "Lumen attention graphs require a Tensor or a (Tensor, None) output"
        )

    def _run_attention_forward(self) -> torch.Tensor:
        return self._primary_output(
            self._forward_attention(self._static_input, **self._static_kwargs)
        )

    def _warmup_on_capture_stream(self, capture_stream: torch.cuda.Stream) -> None:
        """Eager fwd+bwd on the capture stream, matching PyTorch/TE.

        1F1B warmup is not enough: EP overlap and CK FMHA use side streams
        that only get joined to *this* stream after they run while it is
        current. Capturing without that step is what raised
        hipErrorStreamCaptureUnjoined at seq=4096.
        """
        differentiable_inputs = (self._static_input,) + self._parameters
        # One eager pass joins FMHA side streams onto this capture stream.
        # Repeating 1F1B's num_warmup here made 48-layer capture exceed the
        # 10 min NCCL watchdog.
        with torch.cuda.stream(capture_stream), torch.enable_grad():
            output = self._run_attention_forward()
            torch.autograd.grad(
                output,
                differentiable_inputs,
                grad_outputs=torch.ones_like(output),
                retain_graph=False,
                allow_unused=True,
            )

    def _capture(self, hidden_states: torch.Tensor, kwargs: Dict[str, Any]):
        # Snapshot tensor becomes this runner's leaf. Do not clone a second
        # time — that doubled GBS=256 peak memory.
        self._input_signature = _tensor_signature(hidden_states)
        hidden_states.requires_grad_(True)
        self._static_input = hidden_states
        if _share_rope():
            self._static_kwargs = _shared_static_kwargs(self._microbatch, kwargs)
            self._kwarg_signatures = dict(_MB_KWARG_SIGNATURES.get(self._microbatch, {}))
            self._constant_kwargs = dict(_MB_CONSTANT_KWARGS.get(self._microbatch, {}))
        else:
            self._static_kwargs = self._clone_kwargs(kwargs)
        self._pool = self.get_shared_pool()
        device = hidden_states.device
        _synchronize_for_attention_graph_capture(device)
        _clear_async_hip_error()

        capture_stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device)
        default_stream = torch.cuda.default_stream(device)
        capture_stream.wait_stream(current_stream)
        capture_stream.wait_stream(default_stream)

        from lumen.ops.quantize.linear import set_graph_capture_mode

        set_graph_capture_mode(True)
        try:
            self._warmup_on_capture_stream(capture_stream)
            capture_stream.synchronize()
            current_stream.wait_stream(capture_stream)
            default_stream.wait_stream(capture_stream)
            _synchronize_for_attention_graph_capture(device)

            graph_kwargs = {
                "pool": self._pool,
                "stream": capture_stream,
                "capture_error_mode": "thread_local",
            }

            self._fwd_graph = torch.cuda.CUDAGraph()
            with torch.enable_grad(), torch.cuda.graph(
                self._fwd_graph,
                **graph_kwargs,
            ):
                output = self._run_attention_forward()
            self._static_output = output
            # Must be finite: empty_like garbage dY NaNs saved activations
            # that the backward graph then replays forever.
            self._static_grad_output = torch.ones_like(output)

            self._bwd_graph = torch.cuda.CUDAGraph()
            with torch.enable_grad(), torch.cuda.graph(
                self._bwd_graph,
                **graph_kwargs,
            ):
                grads = torch.autograd.grad(
                    self._static_output,
                    (self._static_input,) + self._parameters,
                    grad_outputs=self._static_grad_output,
                    retain_graph=False,
                    allow_unused=True,
                )
            self._static_grad_input = grads[0]
            self._static_parameter_grads = tuple(grads[1:])
            current_stream.wait_stream(capture_stream)
            _synchronize_for_attention_graph_capture(device)
            self._captured = True
        finally:
            set_graph_capture_mode(False)

    def _prepare_capture(self, hidden_states: torch.Tensor, kwargs: Dict[str, Any]):
        """Snapshot an input signature; actual capture runs between train steps."""
        self._capture_input = _schedule_reused_hidden(
            self._layer_number,
            self._microbatch,
            hidden_states,
        )
        if _share_rope():
            self._capture_kwargs = _shared_static_kwargs(self._microbatch, kwargs)
        else:
            self._capture_kwargs = self._clone_kwargs(kwargs)
        self._capture_ready = True
        _PENDING_ATTENTION_GRAPHS.add(self)

    def capture_if_ready(self) -> bool:
        """Capture outside the active 1F1B schedule.

        Snapshot still happens during training; the HIP graph is recorded
        afterwards on a dedicated stream that first replays eager fwd+bwd so
        FMHA/EP side streams are joined before ``cuda.graph``.
        """
        if self._disabled or self._captured or not self._capture_ready:
            return False
        try:
            self._capture(self._capture_input, self._capture_kwargs)
            logger.info("%s fwd/bwd graph capture succeeded", self._name)
            _PENDING_ATTENTION_GRAPHS.discard(self)
            self._capture_ready = False
            self._capture_input = None
            self._capture_kwargs = {}
            return True
        except Exception as error:
            _clear_async_hip_error()
            if _is_retryable_attention_graph_error(error):
                logger.warning(
                    "%s graph capture deferred between train steps: %s; "
                    "will retry after the next train step",
                    self._name,
                    error,
                )
                return False
            logger.warning(
                "%s graph capture failed between train steps: %s; "
                "permanently using eager path",
                self._name,
                error,
            )
            self.reset()
            self._disabled = True
            _PENDING_ATTENTION_GRAPHS.discard(self)
            self._capture_ready = False
            self._capture_input = None
            self._capture_kwargs = {}
            return False

    def _copy_runtime_inputs(self, hidden_states: torch.Tensor):
        if _tensor_signature(hidden_states) != self._input_signature:
            raise RuntimeError(
                f"{self._name} input signature changed after CUDA graph capture"
            )
        _copy_into_static(self._static_input, hidden_states)

    def _validate_kwargs(self, kwargs):
        if set(kwargs) != set(self._static_kwargs):
            raise RuntimeError(f"{self._name} keyword arguments changed after graph capture")
        for key, value in kwargs.items():
            if key in self._kwarg_signatures:
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(f"{self._name} kwarg {key!r} is no longer a tensor")
                if _tensor_signature(value) != self._kwarg_signatures[key]:
                    raise RuntimeError(f"{self._name} kwarg {key!r} signature changed")
                if not _share_rope():
                    _copy_into_static(self._static_kwargs[key], value)
            elif value != self._constant_kwargs[key]:
                raise RuntimeError(f"{self._name} kwarg {key!r} changed after graph capture")

    def __call__(self, *args, **kwargs):
        if self._disabled or not torch.is_grad_enabled():
            return self._forward_attention(*args, **kwargs)

        hidden_states, graph_kwargs = self._split_inputs(args, kwargs)
        if self._captured:
            if not _share_rope():
                self._validate_kwargs(graph_kwargs)
            output = _AttentionGraphReplay.apply(
                self, hidden_states, *self._parameters
            )
            return output, None

        self._call_count += 1
        if (
            self._call_count >= self._num_warmup
            and not self._capture_ready
            and not self._captured
            and _pending_snapshot_count() < _captures_per_step()
            and _layer_in_snapshot_cohort(self._layer_number)
        ):
            self._prepare_capture(hidden_states, graph_kwargs)
        return self._forward_attention(*args, **kwargs)

    def reset(self):
        for graph_name in ("_fwd_graph", "_bwd_graph"):
            graph = getattr(self, graph_name)
            if graph is not None and hasattr(graph, "reset"):
                try:
                    graph.reset()
                except Exception:
                    # A failed capture may leave an invalid graph handle. There
                    # is nothing useful to reset in that case.
                    pass
            setattr(self, graph_name, None)
        self._captured = False
        self._static_input = None
        self._static_kwargs = {}
        self._static_output = None
        self._static_grad_output = None
        self._static_grad_input = None
        self._static_parameter_grads = ()
        self._capture_ready = False
        self._capture_input = None
        self._capture_kwargs = {}


class LumenAttentionGraphDispatcher:
    """Dispatch one attention graph runner per microbatch (TE-style).

    Each microbatch keeps its own graph executable while compatible static
    buffers are reused through the shared graph pool. Set
    ``LUMEN_ATTN_GRAPH_PER_MICROBATCH=0`` to force a single runner per layer.
    """

    def __init__(
        self,
        layer: nn.Module,
        *,
        num_warmup: int = 3,
        max_microbatches: int = 0,
    ):
        self.layer = layer
        self.original_forward_attention = layer._forward_attention
        self.num_warmup = num_warmup
        self.max_microbatches = max_microbatches
        self.per_microbatch = os.environ.get("LUMEN_ATTN_GRAPH_PER_MICROBATCH", "1") != "0"
        self.runners: Dict[int, LumenGraphedAttention] = {}
        self.parameters = self._collect_attention_parameters(layer)
        self.manual_hooks = []

    @staticmethod
    def _collect_attention_parameters(layer) -> Tuple[torch.nn.Parameter, ...]:
        modules = [
            getattr(layer, "input_layernorm", None),
            getattr(layer, "self_attention", None),
            getattr(layer, "pre_cross_attn_layernorm", None),
            getattr(layer, "cross_attention", None),
        ]
        seen, parameters = set(), []
        for module in modules:
            if not isinstance(module, nn.Module):
                continue
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return tuple(parameters)

    def __call__(self, *args, **kwargs):
        microbatch = int(getattr(self.layer, "current_microbatch", 0))
        if self.max_microbatches > 0 and microbatch >= self.max_microbatches:
            return self.original_forward_attention(*args, **kwargs)
        if not self.per_microbatch:
            microbatch = 0
        runner = self.runners.get(microbatch)
        if runner is None:
            layer_number = int(getattr(self.layer, "layer_number", 0) or 0)
            runner = LumenGraphedAttention(
                self.original_forward_attention,
                self.parameters,
                num_warmup=self.num_warmup,
                microbatch=microbatch,
                layer_number=layer_number,
                name=f"layer {layer_number} microbatch {microbatch} attention",
            )
            self.runners[microbatch] = runner
        if runner._captured:
            if os.environ.get("LUMEN_ATTN_GRAPH_MANUAL_HOOKS", "1") != "0":
                for hook, hook_args in self.manual_hooks:
                    hook(*hook_args)
        return runner(*args, **kwargs)


def _collect_attention_manual_hooks(
    layer: nn.Module,
    make_forward_pre_hook: Callable,
) -> List[Tuple[Callable, Tuple[nn.Module, ...]]]:
    """Collect DDP manual hooks for attention-only graphs.

    ``GraphableMegatronModule.setup_manual_hooks`` uses
    ``_get_submodules_under_cudagraphs()``. With TE attn scope that list is
    limited to LN/attention modules. Lumen installs graphs with
    ``cuda_graph_impl=none``, so Megatron leaves ``cuda_graph_scope`` empty and
    the default path hooks *every* parameter module in the layer — including
    MoE/MLP. Calling ``finish_param_sync`` for those buckets before each
    attention replay blocks the CPU on ``hipEventSynchronize`` and kills EP /
    param-gather overlap.
    """
    try:
        from megatron.core.transformer.identity_op import IdentityOp
    except Exception:
        IdentityOp = ()  # type: ignore[misc, assignment]

    submodules = [
        getattr(layer, "input_layernorm", None),
        getattr(layer, "self_attention", None),
        getattr(layer, "pre_cross_attn_layernorm", None),
        getattr(layer, "cross_attention", None),
    ]
    param_modules: Dict[int, nn.Module] = {}
    for submodule in submodules:
        if submodule is None or isinstance(submodule, IdentityOp):
            continue
        for module in submodule.modules():
            if next(module.parameters(recurse=False), None) is not None:
                param_modules[id(module)] = module
    return [(make_forward_pre_hook(), (module,)) for module in param_modules.values()]


def install_attention_graph_capture(
    model: nn.Module,
    *,
    num_warmup: int = 3,
    skip_recomputed_layers: int = 0,
    max_graphed_layers: int = 0,
    max_microbatches: int = 0,
    make_forward_pre_hook: Optional[Callable] = None,
) -> int:
    """Install Lumen-native attn-only fwd/bwd graph dispatchers."""
    global _ATTENTION_GRAPH_TOTAL_LAYERS

    if not hasattr(model, "decoder") or not hasattr(model.decoder, "layers"):
        logger.warning("Model has no decoder layers; skipping attention graph wrappers")
        return 0

    _ATTENTION_GRAPH_TOTAL_LAYERS = len(model.decoder.layers)
    wrapped = 0
    for layer_index, layer in enumerate(model.decoder.layers):
        if layer_index < skip_recomputed_layers:
            continue
        if max_graphed_layers > 0 and wrapped >= max_graphed_layers:
            break
        if not hasattr(layer, "_forward_attention"):
            continue
        dispatcher = LumenAttentionGraphDispatcher(
            layer,
            num_warmup=num_warmup,
            max_microbatches=max_microbatches,
        )
        layer._lumen_attention_graph_dispatcher = dispatcher
        layer._forward_attention = functools.wraps(dispatcher.original_forward_attention)(
            dispatcher
        )
        if make_forward_pre_hook is not None:
            dispatcher.manual_hooks = _collect_attention_manual_hooks(
                layer, make_forward_pre_hook
            )
        wrapped += 1

    logger.info(
        "Installed Lumen attention graph capture on %d transformer layers", wrapped
    )
    return wrapped


def _capture_attention_batch(
    pending: List[LumenGraphedAttention],
) -> int:
    """Capture one schedule-ordered batch using TE's allocation protocol.

    Capture order follows Megatron's real PP=1 schedule. With EP overlap, mb0
    first runs forward alone, then ``F(mb+1, layer)`` is interleaved with
    ``B(mb, reverse_layer)``. TE constructs the same layer-expanded ``_order``.
    """
    if not pending:
        return 0

    pool = LumenGraphedAttention.get_shared_pool()
    device = pending[0]._capture_input.device
    capture_stream = torch.cuda.Stream(device=device)
    graph_kwargs = {
        "pool": pool,
    }

    # Adopt snapshots as static inputs without another hidden-state clone.
    for runner in pending:
        hidden_states = runner._capture_input
        kwargs = runner._capture_kwargs
        runner._input_signature = _tensor_signature(hidden_states)
        hidden_states.requires_grad_(True)
        runner._static_input = hidden_states
        runner._static_kwargs = kwargs
        runner._kwarg_signatures = {
            key: _tensor_signature(value)
            for key, value in kwargs.items()
            if isinstance(value, torch.Tensor)
        }
        runner._constant_kwargs = {
            key: value
            for key, value in kwargs.items()
            if not isinstance(value, torch.Tensor)
        }
        runner._pool = pool

    _synchronize_for_attention_graph_capture(device)
    _clear_async_hip_error()
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    capture_stream.wait_stream(torch.cuda.default_stream(device))

    from lumen.ops.quantize.linear import set_graph_capture_mode

    set_graph_capture_mode(True)
    try:
        # TE warms every callable on one side stream before any capture.
        for runner in pending:
            runner._warmup_on_capture_stream(capture_stream)
        capture_stream.synchronize()
        _synchronize_for_attention_graph_capture(device)

        runner_map = {
            (runner._microbatch, runner._layer_number): runner
            for runner in pending
        }
        microbatches = sorted({runner._microbatch for runner in pending})
        total_layers = max(
            _ATTENTION_GRAPH_TOTAL_LAYERS,
            max(runner._layer_number for runner in pending),
        )

        def capture_forward(runner):
            runner._fwd_graph = torch.cuda.CUDAGraph()
            with torch.enable_grad(), torch.cuda.graph(
                runner._fwd_graph,
                **graph_kwargs,
            ):
                runner._static_output = runner._run_attention_forward()

        previous_backward_runner = None
        shared_grad_outputs: Dict[Tuple[Any, ...], torch.Tensor] = {}

        def capture_backward(runner):
            nonlocal previous_backward_runner
            output_key = _tensor_signature(runner._static_output)
            runner._static_grad_output = shared_grad_outputs.get(output_key)
            if runner._static_grad_output is None:
                runner._static_grad_output = torch.ones_like(runner._static_output)
                shared_grad_outputs[output_key] = runner._static_grad_output
            runner._bwd_graph = torch.cuda.CUDAGraph()
            with torch.enable_grad(), torch.cuda.graph(
                runner._bwd_graph,
                **graph_kwargs,
            ):
                grads = torch.autograd.grad(
                    runner._static_output,
                    (runner._static_input,) + runner._parameters,
                    grad_outputs=runner._static_grad_output,
                    only_inputs=True,
                    allow_unused=True,
                    retain_graph=False,
                )
            runner._static_grad_input = grads[0]
            runner._static_parameter_grads = tuple(grads[1:])
            runner._static_output = _make_weak_tensor(runner._static_output)

            if previous_backward_runner is not None:
                previous_backward_runner._static_grad_input = _make_weak_tensor(
                    previous_backward_runner._static_grad_input
                )
                previous_backward_runner._static_parameter_grads = tuple(
                    _make_weak_tensor(grad)
                    for grad in previous_backward_runner._static_parameter_grads
                )
            previous_backward_runner = runner

        try:
            from megatron.training import get_args

            ep_overlap = bool(
                getattr(get_args(), "overlap_moe_expert_parallel_comm", False)
            )
        except Exception:
            ep_overlap = False

        if ep_overlap and microbatches:
            first_mb = microbatches[0]
            for layer_number in range(1, total_layers + 1):
                runner = runner_map.get((first_mb, layer_number))
                if runner is not None:
                    capture_forward(runner)

            for previous_mb, forward_mb in zip(microbatches, microbatches[1:]):
                for layer_number in range(1, total_layers + 1):
                    runner = runner_map.get((forward_mb, layer_number))
                    if runner is not None:
                        capture_forward(runner)
                    backward_layer = total_layers - layer_number + 1
                    runner = runner_map.get((previous_mb, backward_layer))
                    if runner is not None:
                        capture_backward(runner)

            last_mb = microbatches[-1]
            for layer_number in range(total_layers, 0, -1):
                runner = runner_map.get((last_mb, layer_number))
                if runner is not None:
                    capture_backward(runner)
        else:
            for microbatch in microbatches:
                layer_runners = sorted(
                    (
                        runner
                        for runner in pending
                        if runner._microbatch == microbatch
                    ),
                    key=lambda runner: runner._layer_number,
                )
                for runner in layer_runners:
                    capture_forward(runner)
                for runner in reversed(layer_runners):
                    capture_backward(runner)

        capture_stream.synchronize()
        _synchronize_for_attention_graph_capture(device)
    finally:
        set_graph_capture_mode(False)

    for runner in pending:
        runner._captured = True
        runner._capture_ready = False
        runner._capture_input = None
        runner._capture_kwargs = {}
        _PENDING_ATTENTION_GRAPHS.discard(runner)
        logger.info("%s fwd/bwd graph capture succeeded", runner._name)
    _log_static_buffer_reuse(pending)
    return len(pending)


def _log_static_buffer_reuse(runners: Sequence["LumenGraphedAttention"]) -> None:
    """Report how well graph IO buffers are shared across captured pairs.

    Distinct-address counts well below the runner count mean the graph pool
    handed the same block to later captures. Memory-bound eager kernels that
    read these buffers degrade sharply when the working set is not reused.
    """

    def survey(attr: str) -> str:
        addresses, total_bytes = {}, 0
        for runner in runners:
            tensor = getattr(runner, attr, None)
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                continue
            nbytes = tensor.numel() * tensor.element_size()
            if tensor.data_ptr() not in addresses:
                total_bytes += nbytes
            addresses[tensor.data_ptr()] = nbytes
        return (
            f"{attr}: {len(addresses)} distinct / {len(runners)} runners, "
            f"{total_bytes / 2**30:.2f} GiB"
        )

    logger.info(
        "Lumen attention graph static buffers -> %s | %s | %s",
        survey("_static_input"),
        survey("_static_output"),
        survey("_static_grad_input"),
    )


def capture_pending_attention_graphs(max_captures: int = None) -> int:
    """Capture a bounded, schedule-ordered batch at a train-step boundary."""
    pending = sorted(
        _PENDING_ATTENTION_GRAPHS,
        key=lambda runner: (
            int(getattr(runner, "_microbatch", 0)),
            int(getattr(runner, "_layer_number", 10**9)),
        ),
    )
    if not pending:
        return 0
    if max_captures is None:
        max_captures = _captures_per_step()
    if max_captures > 0:
        pending = pending[:max_captures]
    try:
        captured = _capture_attention_batch(pending)
    except Exception as error:
        _clear_async_hip_error()
        logger.warning(
            "Lumen attention graph batch capture failed: %s; deferring batch",
            error,
        )
        return 0

    remaining = len(_PENDING_ATTENTION_GRAPHS)
    if remaining:
        logger.info(
            "Lumen attention graphs: captured %d this step, %d still pending",
            captured,
            remaining,
        )
    return captured
