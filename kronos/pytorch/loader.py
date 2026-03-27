# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kronos model loader implementation for financial time series forecasting.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from .src.model import Kronos


@dataclass
class KronosConfig(ModelConfig):
    context_length: int = 512
    s1_bits: int = 10
    s2_bits: int = 10


class ModelVariant(StrEnum):
    SMALL = "small"


class ModelLoader(ForgeModel):
    """Kronos model loader for financial time series forecasting.

    Loads the Kronos decoder-only Transformer model that operates on
    hierarchical discrete tokens from OHLCV candlestick data.
    """

    _VARIANTS = {
        ModelVariant.SMALL: KronosConfig(
            pretrained_model_name="NeoQuasar/Kronos-small",
            context_length=512,
            s1_bits=10,
            s2_bits=10,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Kronos",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Kronos model for financial time series forecasting.

        Returns:
            torch.nn.Module: The Kronos model instance.
        """
        cfg = self._variant_config

        model = Kronos.from_pretrained(cfg.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample token inputs for the Kronos model.

        Returns:
            list: A list of [s1_ids, s2_ids] integer tensors,
                  each of shape (batch_size, seq_len).
        """
        cfg = self._variant_config
        seq_len = 64
        s1_vocab_size = 2**cfg.s1_bits
        s2_vocab_size = 2**cfg.s2_bits

        torch.manual_seed(42)
        s1_ids = torch.randint(0, s1_vocab_size, (1, seq_len))
        s2_ids = torch.randint(0, s2_vocab_size, (1, seq_len))

        return [s1_ids, s2_ids]
