# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kronos model loader implementation for financial time series forecasting.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from model import Kronos, KronosTokenizer

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


@dataclass
class KronosConfig(ModelConfig):
    context_length: int = 2048
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k"


class ModelVariant(StrEnum):
    MINI = "mini"


class ModelLoader(ForgeModel):
    """Kronos model loader for financial time series forecasting.

    Loads the Kronos autoregressive transformer model pre-trained
    on discretized K-line (candlestick) data for zero-shot
    financial time series forecasting.
    """

    _VARIANTS = {
        ModelVariant.MINI: KronosConfig(
            pretrained_model_name="NeoQuasar/Kronos-mini",
            context_length=2048,
            tokenizer_name="NeoQuasar/Kronos-Tokenizer-2k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINI

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

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
            torch.nn.Module: The Kronos autoregressive transformer model.
        """
        cfg = self._variant_config

        model = Kronos.from_pretrained(cfg.pretrained_model_name)
        self._tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Generates synthetic OHLCV candlestick data and tokenizes it
        using the Kronos tokenizer.

        Returns:
            dict: Tokenized input tensors for the model.
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        context = torch.randn(1, cfg.context_length, dtype=dtype)

        if self._tokenizer is not None:
            token_ids, attention_mask, _ = self._tokenizer.context_input_transform(
                context
            )
            return {"input_ids": token_ids, "attention_mask": attention_mask}

        return {"input_ids": context}
