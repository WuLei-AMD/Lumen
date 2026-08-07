"""DeepSeek-V4 (DSV4) model integration for Lumen (no Transformer Engine)."""

from lumen.models.dsv4.compressor import DeepSeekV4Compressor
from lumen.models.dsv4.hyper_connection import (
    DeepSeekV4HyperConnectionUtil,
    HCHeadParams,
)
from lumen.models.dsv4.megatron.spec import get_dsv4_spec

__all__ = [
    "DeepSeekV4Compressor",
    "DeepSeekV4HyperConnectionUtil",
    "HCHeadParams",
    "get_dsv4_spec",
]
