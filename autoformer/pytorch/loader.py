# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Autoformer model loader implementation for time series forecasting.
"""

from typing import Optional

import torch
from transformers import AutoformerConfig, AutoformerForPrediction

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
    """Available Autoformer model variants."""

    TOURISM_MONTHLY = "Tourism_Monthly"


class ModelLoader(ForgeModel):
    """Autoformer model loader for time series forecasting."""

    _VARIANTS = {
        ModelVariant.TOURISM_MONTHLY: ModelConfig(
            pretrained_model_name="huggingface/autoformer-tourism-monthly",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TOURISM_MONTHLY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Autoformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Autoformer model."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoformerForPrediction.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the Autoformer model.

        Returns:
            dict: Input dict with past_values, past_time_features,
                  past_observed_mask, static_categorical_features,
                  future_values, and future_time_features tensors.
        """
        dtype = dtype_override or torch.float32
        torch.manual_seed(42)

        # The model requires context_length + max(lags_sequence) past time steps
        # For tourism-monthly: context_length=24, max(lags_sequence)=37, so past_length=61
        # num_time_features=2, num_static_categorical_features=1, prediction_length=24
        config = AutoformerConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        past_length = config.context_length + max(config.lags_sequence)
        prediction_length = config.prediction_length
        num_time_features = config.num_time_features

        past_values = torch.randn(1, past_length, dtype=dtype)
        past_time_features = torch.randn(1, past_length, num_time_features, dtype=dtype)
        past_observed_mask = torch.ones(1, past_length, dtype=dtype)
        static_categorical_features = torch.zeros(1, 1, dtype=torch.long)
        future_values = torch.randn(1, prediction_length, dtype=dtype)
        future_time_features = torch.randn(
            1, prediction_length, num_time_features, dtype=dtype
        )

        return {
            "past_values": past_values,
            "past_time_features": past_time_features,
            "past_observed_mask": past_observed_mask,
            "static_categorical_features": static_categorical_features,
            "future_values": future_values,
            "future_time_features": future_time_features,
        }
