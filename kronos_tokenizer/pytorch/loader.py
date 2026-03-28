# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kronos-Tokenizer-2k model loader for financial time-series tokenization.

Kronos-Tokenizer-2k is a hierarchical encoder-decoder tokenizer that quantizes
continuous multi-dimensional financial candlestick (OHLCV) data into discrete tokens.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from transformers import AutoModel

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
class KronosTokenizerConfig(ModelConfig):
    group_size: int = 5
    d_in: int = 6


class ModelVariant(StrEnum):
    TOKENIZER_2K = "tokenizer-2k"


class ModelLoader(ForgeModel):
    """Kronos-Tokenizer-2k model loader for financial time-series tokenization.

    Loads the Kronos hierarchical tokenizer that encodes OHLCV candlestick data
    into discrete tokens for downstream financial forecasting models.
    """

    _VARIANTS = {
        ModelVariant.TOKENIZER_2K: KronosTokenizerConfig(
            pretrained_model_name="NeoQuasar/Kronos-Tokenizer-2k",
            group_size=5,
            d_in=6,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TOKENIZER_2K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="KronosTokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Kronos-Tokenizer-2k model.

        Returns:
            torch.nn.Module: The Kronos tokenizer model instance.
        """
        cfg = self._variant_config

        model = AutoModel.from_pretrained(
            cfg.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=dtype_override or torch.float32,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample OHLCV candlestick inputs for the model.

        Returns:
            torch.Tensor: Input tensor of shape (batch_size, seq_len, d_in)
                where d_in=6 corresponds to Open, High, Low, Close, Volume, Amount.
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        # Generate synthetic OHLCV data: (batch_size, num_groups * group_size, d_in)
        torch.manual_seed(42)
        seq_len = cfg.group_size * 4  # 4 groups of candlesticks
        inputs = torch.randn(1, seq_len, cfg.d_in, dtype=dtype)

        return inputs
