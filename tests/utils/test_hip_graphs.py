###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import torch
import torch.nn as nn
import pytest


class TestLumenGraphedCallable:

    def test_construction(self):
        from lumen.utils.hip_graphs import LumenGraphedCallable

        def fn(x):
            return x * 2

        sample = torch.randn(4, 4)
        # Won't capture without CUDA, but should not crash
        gc = LumenGraphedCallable(fn, (sample,))
        assert gc is not None

    def test_call_without_cuda(self):
        from lumen.utils.hip_graphs import LumenGraphedCallable

        def fn(x):
            return x * 2

        sample = torch.randn(4, 4)
        gc = LumenGraphedCallable(fn, (sample,))
        result = gc(torch.ones(4, 4))
        expected = torch.ones(4, 4) * 2
        torch.testing.assert_close(result, expected)

    def test_reset(self):
        from lumen.utils.hip_graphs import LumenGraphedCallable

        def fn(x):
            return x + 1

        gc = LumenGraphedCallable(fn, (torch.randn(4, 4),))
        gc.reset()
        assert gc._graph is None


class TestLumenGraphedModule:

    def test_construction(self):
        from lumen.utils.hip_graphs import LumenGraphedModule

        module = nn.Linear(8, 4)
        gm = LumenGraphedModule(module, enabled=False)
        assert gm.module is module

    def test_forward_disabled(self):
        from lumen.utils.hip_graphs import LumenGraphedModule

        module = nn.Linear(8, 4)
        gm = LumenGraphedModule(module, enabled=False)
        x = torch.randn(2, 8)
        out = gm(x)
        assert out.shape == (2, 4)

    def test_release_graph(self):
        from lumen.utils.hip_graphs import LumenGraphedModule

        module = nn.Linear(8, 4)
        gm = LumenGraphedModule(module, enabled=False)
        gm.release_graph()
        assert gm._graphed is None


class TestMakeGraphedCallables:

    def test_multiple_callables(self):
        from lumen.utils.hip_graphs import lumen_make_graphed_callables

        def fn1(x):
            return x * 2

        def fn2(x):
            return x + 1

        args1 = (torch.randn(4, 4),)
        args2 = (torch.randn(4, 4),)
        result = lumen_make_graphed_callables([fn1, fn2], [args1, args2])
        assert len(result) == 2


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward_attention(self, hidden_states, attention_mask=None):
        del attention_mask
        return torch.relu(self.proj(hidden_states)), None


class TestLumenGraphedAttention:
    def test_requires_cuda_input(self):
        from lumen.utils.hip_graphs import LumenGraphedAttention

        module = _TinyAttention()
        graph = LumenGraphedAttention(
            module.forward_attention, module.parameters(), num_warmup=1
        )
        with pytest.raises(RuntimeError, match="CUDA hidden_states"):
            graph(torch.randn(2, 8))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
    def test_forward_backward_matches_eager(self):
        from lumen.utils.hip_graphs import (
            LumenGraphedAttention,
            capture_pending_attention_graphs,
        )

        torch.manual_seed(123)
        eager = _TinyAttention().cuda()
        graphed_module = _TinyAttention().cuda()
        graphed_module.load_state_dict(eager.state_dict())
        graphed = LumenGraphedAttention(
            graphed_module.forward_attention,
            graphed_module.parameters(),
            num_warmup=1,
        )

        # First call warms kernels and allocators.
        warmup = torch.randn(4, 8, device="cuda", requires_grad=True)
        graphed(warmup)[0].sum().backward()
        graphed_module.zero_grad(set_to_none=True)
        assert capture_pending_attention_graphs() == 1

        # Exercise repeated replay so a later backward overwrites the static
        # grad surface only after AccumulateGrad consumed the previous result.
        for _ in range(3):
            eager.zero_grad(set_to_none=True)
            graphed_module.zero_grad(set_to_none=True)
            eager_input = torch.randn(4, 8, device="cuda", requires_grad=True)
            graph_input = eager_input.detach().clone().requires_grad_(True)
            eager_output = eager.forward_attention(eager_input)[0]
            graph_output = graphed(graph_input)[0]
            torch.testing.assert_close(graph_output, eager_output)

            eager_output.square().sum().backward()
            graph_output.square().sum().backward()
            torch.testing.assert_close(graph_input.grad, eager_input.grad)
            torch.testing.assert_close(
                graphed_module.proj.weight.grad, eager.proj.weight.grad
            )
        assert graphed.captured
