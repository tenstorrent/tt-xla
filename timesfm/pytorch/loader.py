# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TimesFM 2.5 model loader implementation for time series forecasting.
"""

from typing import Optional

import torch
from transformers import TimesFm2_5ModelForPrediction

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


class ModelVariant(StrEnum):
    """Available TimesFM 2.5 model variants."""

    TIMESFM_2_5_200M = "TimesFM_2_5_200M"


class ModelLoader(ForgeModel):
    """TimesFM 2.5 model loader for time series forecasting."""

    _VARIANTS = {
        ModelVariant.TIMESFM_2_5_200M: ModelConfig(
            pretrained_model_name="google/timesfm-2.5-200m-transformers",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TIMESFM_2_5_200M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TimesFM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TimesFM 2.5 model."""
        model = TimesFm2_5ModelForPrediction.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype_override or torch.float32,
            **kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'past_values' list of tensors and 'forecast_context_len'.
        """
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        past_values = [
            torch.randn(100, dtype=dtype),
        ]

        return {
            "past_values": past_values,
            "forecast_context_len": 512,
        }
