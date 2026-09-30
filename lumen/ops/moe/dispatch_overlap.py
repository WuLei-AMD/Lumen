###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Communication helpers for overlapping MoE token dispatch metadata.

Variable-size all-to-all requires host split lists.  The count exchange can,
however, run asynchronously while the caller sorts tokens and builds the data
payload.  This mirrors Megatron's deferred synchronization without coupling the
operation to a specific model implementation.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist


@dataclass
class AsyncCountExchange:
    """Pending non-differentiable MoE count exchange."""

    send_counts: torch.Tensor
    recv_counts: torch.Tensor
    work: Optional[Any] = None

    def wait_for_splits(self) -> tuple[list[int], list[int]]:
        """Wait for communication and return host split-size lists."""
        if self.work is not None:
            self.work.wait()
        return self.send_counts.tolist(), self.recv_counts.tolist()


def begin_count_exchange(
    send_counts: torch.Tensor,
    group: Optional[dist.ProcessGroup],
) -> AsyncCountExchange:
    """Start an asynchronous all-to-all exchange of per-rank token counts.

    The returned object owns both tensors until :meth:`wait_for_splits` is
    called, so the caller can safely perform token permutation and payload
    construction while the collective is in flight.
    """
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return AsyncCountExchange(send_counts, send_counts)

    recv_counts = torch.empty_like(send_counts)
    work = dist.all_to_all_single(
        recv_counts,
        send_counts,
        group=group,
        async_op=True,
    )
    return AsyncCountExchange(send_counts, recv_counts, work)


_SIDE_STREAM: Optional[torch.cuda.Stream] = None
_PINNED: dict[tuple, torch.Tensor] = {}
_DEVICE_A2A_LOGGED = False


def device_a2a_enabled() -> bool:
    """Match Megatron: on unless ``LUMEN_MOE_DEVICE_A2A=0``."""
    import os

    return os.environ.get("LUMEN_MOE_DEVICE_A2A", "1") != "0"


def _side_stream() -> torch.cuda.Stream:
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


def _pinned(slot: str, flat: torch.Tensor) -> torch.Tensor:
    device_index = flat.device.index if flat.device.index is not None else -1
    key = (slot, device_index, flat.numel(), flat.dtype)
    cached = _PINNED.get(key)
    if cached is None:
        cached = torch.empty(
            flat.numel(),
            dtype=flat.dtype,
            device="cpu",
            pin_memory=True,
        )
        _PINNED[key] = cached
    return cached


def host_split_pair(
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
) -> tuple[list[int], list[int]]:
    """Return host split lists without draining the compute stream.

    RCCL still reads split sizes on the host. ``Tensor.tolist()`` synchronizes
    the current stream first, so a token gather queued there cannot overlap
    the copy. Sum and D2H run on a side stream that does not wait for that
    gather. A 1-D tensor is copied as-is; a 2-D tensor is summed over its
    last dimension.
    """
    if not device_a2a_enabled() or not send_counts.is_cuda:
        send_flat = send_counts if send_counts.dim() == 1 else send_counts.sum(dim=-1)
        recv_flat = recv_counts if recv_counts.dim() == 1 else recv_counts.sum(dim=-1)
        return send_flat.tolist(), recv_flat.tolist()

    global _DEVICE_A2A_LOGGED
    if not _DEVICE_A2A_LOGGED:
        _DEVICE_A2A_LOGGED = True
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                "FSDP MoE device all-to-all: split sizes D2H on a side stream",
                flush=True,
            )

    stream = _side_stream()
    send_counts.record_stream(stream)
    recv_counts.record_stream(stream)
    with torch.cuda.stream(stream):
        send_flat = send_counts if send_counts.dim() == 1 else send_counts.sum(dim=-1)
        recv_flat = recv_counts if recv_counts.dim() == 1 else recv_counts.sum(dim=-1)
        send_flat = send_flat.contiguous().reshape(-1)
        recv_flat = recv_flat.contiguous().reshape(-1)
        send_host = _pinned("send", send_flat)
        recv_host = _pinned("recv", recv_flat)
        send_host.copy_(send_flat, non_blocking=True)
        recv_host.copy_(recv_flat, non_blocking=True)
        send_flat.record_stream(stream)
        recv_flat.record_stream(stream)
    event = stream.record_event()
    if not event.query():
        event.synchronize()
    return (
        [int(value) for value in send_host.tolist()],
        [int(value) for value in recv_host.tolist()],
    )


_COMM_STREAMS: dict[int, torch.cuda.Stream] = {}


class PendingPayloadA2A:
    """One payload all-to-all on the EP stream.

    ``done`` is recorded on that stream immediately after this collective is
    queued. Waiting on ``done`` does not wait for a later collective on the
    same stream, and it does not form a cycle with ``wait_stream``.
    """

    def __init__(
        self,
        output: torch.Tensor,
        work: Optional[Any],
        done: Optional[torch.cuda.Event],
        source: Optional[torch.Tensor] = None,
    ):
        self.output = output
        self.work = work
        self.done = done
        self.source = source

    def wait_work(self) -> None:
        """Block the host until this collective finishes. The group is then free."""
        if self.work is not None:
            self.work.wait()

    def wait(self) -> torch.Tensor:
        self.wait_work()
        if self.done is not None:
            torch.cuda.current_stream().wait_event(self.done)
        self.source = None
        return self.output


def _comm_stream(device_index: int) -> torch.cuda.Stream:
    stream = _COMM_STREAMS.get(device_index)
    if stream is None:
        # Higher priority so the payload kernel is queued ahead of a full-grid GEMM.
        try:
            priority = torch.cuda.Stream.priority_range()[1]
        except Exception:
            priority = 0
        stream = torch.cuda.Stream(device=device_index, priority=priority)
        _COMM_STREAMS[device_index] = stream
    return stream


def launch_payload_all_to_all(
    tensor: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    group: Optional[dist.ProcessGroup],
    ready_event: torch.cuda.Event,
) -> PendingPayloadA2A:
    """Start a payload all-to-all that waits for ``ready_event`` and nothing later.

    The collective runs on its own stream, so a later kernel on the compute
    stream does not delay it. Caller must ``wait`` before launching another
    collective on ``group``.
    """
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return PendingPayloadA2A(tensor, None, None)

    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _comm_stream(device_index)
    stream.wait_event(ready_event)
    # The payload is read on this stream while the compute stream allocates the
    # expert workspace. record_stream keeps the caching allocator from handing
    # that storage out until the collective's stream has retired it.
    tensor.record_stream(stream)
    with torch.cuda.stream(stream):
        source = tensor.contiguous()
        output = source.new_empty((sum(recv_splits), *source.shape[1:]))
        work = dist.all_to_all_single(
            output,
            source,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=group,
            async_op=True,
        )
        done = stream.record_event()
        source.record_stream(stream)
        output.record_stream(stream)
    return PendingPayloadA2A(output, work, done, source)
