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
