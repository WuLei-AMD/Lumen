"""Lumen backend spec provider for DSV4 transformer blocks (MoE/MLP, no TE)."""

import os
from typing import Optional, Tuple

from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP

from lumen.models.dsv4.megatron.layers import (
    LumenColumnParallelGroupedLinear,
    LumenColumnParallelLinear,
    LumenRowParallelGroupedLinear,
    LumenRowParallelLinear,
    LocalRMSNorm,
    _dsv4_use_local_rmsnorm,
)
from lumen.models.spec_provider import LumenSpecProvider


class LumenDSV4SpecProvider(LumenSpecProvider):
    """Backend for DSV4 MoE/MLP/dense layers — Lumen linear/norm with tuned BF16 GEMM."""

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False):
        if _dsv4_use_local_rmsnorm():
            return LocalRMSNorm
        return super().layer_norm(rms_norm, for_qk)

    def column_parallel_linear(self):
        return LumenColumnParallelLinear

    def row_parallel_linear(self):
        return LumenRowParallelLinear

    def grouped_mlp_modules(
        self,
        moe_use_grouped_gemm: bool = False,
        moe_use_legacy_grouped_gemm: bool = False,
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        if moe_use_grouped_gemm and not moe_use_legacy_grouped_gemm:
            return TEGroupedMLP, MLPSubmodules(
                linear_fc1=LumenColumnParallelGroupedLinear,
                linear_fc2=LumenRowParallelGroupedLinear,
            )

        return SequentialMLP, MLPSubmodules(
            linear_fc1=self.column_parallel_linear(),
            linear_fc2=self.row_parallel_linear(),
        )
