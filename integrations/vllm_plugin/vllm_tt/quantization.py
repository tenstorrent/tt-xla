# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

"""TT-compatible quantization configs.

Provides ``Mxfp4Config(dequantize=True)`` which tells vllm to use
the mxfp4 quantization-aware weight-loading path but replace every
quantized kernel with its unquantized (bfloat16) equivalent.  This is
the recommended way to run models whose checkpoints ship quantized
weights (e.g. DeepSeek-V3 fp8) on TT hardware that does not support
quantized matmuls natively.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.quantization.mxfp4 import (
    Mxfp4Config as _VllmMxfp4Config,
)


class Mxfp4Config(_VllmMxfp4Config):
    """Extended :class:`Mxfp4Config` with ``dequantize`` support.

    Parameters
    ----------
    dequantize : bool
        When *True*, :meth:`get_quant_method` returns
        ``UnquantizedLinearMethod`` for linear layers and ``None`` for
        all other layer types (MoE, Attention, …), causing every layer
        to use bfloat16 computation.  Raw fp8 parameters are converted
        to bfloat16 during model loading by the TT model runner.
    ignored_layers : list[str] | None
        Forwarded to the base :class:`Mxfp4Config`.
    """

    def __init__(
        self,
        dequantize: bool = False,
        ignored_layers: list[str] | None = None,
    ):
        super().__init__(ignored_layers=ignored_layers)
        self.dequantize = dequantize

    # ------------------------------------------------------------------
    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # TT does not use CUDA capability; return 0 so the capability
        # check (when non-None) never rejects.
        return 0

    # ------------------------------------------------------------------
    def get_quant_method(self, layer, prefix):  # type: ignore[override]
        if self.dequantize:
            from vllm.model_executor.layers.linear import (
                LinearBase,
                UnquantizedLinearMethod,
            )

            # LinearBase asserts quant_method is not None, so we must
            # return an explicit UnquantizedLinearMethod for it.
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            # Everything else (FusedMoE, Attention, …) gets None which
            # makes the layer use its own default unquantized path.
            return None
        return super().get_quant_method(layer, prefix)
