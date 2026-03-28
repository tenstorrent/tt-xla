# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Informer model loader implementation for time series forecasting.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from transformers import InformerForPrediction

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
class InformerModelConfig(ModelConfig):
    context_length: int = 24
    prediction_length: int = 24
    lags_sequence_length: int = 37
    num_time_features: int = 2
    num_static_categorical_features: int = 1


class ModelVariant(StrEnum):
    TOURISM_MONTHLY = "tourism_monthly"


class ModelLoader(ForgeModel):
    """Informer model loader for time series forecasting.

    Loads the Informer encoder-decoder transformer model with
    ProbSparse self-attention for long sequence time series forecasting.
    """

    _VARIANTS = {
        ModelVariant.TOURISM_MONTHLY: InformerModelConfig(
            pretrained_model_name="huggingface/informer-tourism-monthly",
            context_length=24,
            prediction_length=24,
            lags_sequence_length=37,
            num_time_features=2,
            num_static_categorical_features=1,
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
            model="Informer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Informer model for time series forecasting.

        Returns:
            torch.nn.Module: The InformerForPrediction model instance.
        """
        cfg = self._variant_config

        model = InformerForPrediction.from_pretrained(
            cfg.pretrained_model_name,
            torch_dtype=dtype_override or torch.float32,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the Informer model.

        Returns:
            dict: Input tensors matching InformerForPrediction.forward signature.
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)

        # Sequence length = context_length + max(lags_sequence)
        sequence_length = cfg.context_length + cfg.lags_sequence_length

        # past_values: (batch, sequence_length)
        past_values = torch.randn(1, sequence_length, dtype=dtype)

        # past_time_features: (batch, sequence_length, num_time_features)
        past_time_features = torch.randn(
            1, sequence_length, cfg.num_time_features, dtype=dtype
        )

        # past_observed_mask: (batch, sequence_length)
        past_observed_mask = torch.ones(1, sequence_length, dtype=dtype)

        # static_categorical_features: (batch, num_static_categorical_features)
        static_categorical_features = torch.zeros(
            1, cfg.num_static_categorical_features, dtype=torch.long
        )

        # future_time_features: (batch, prediction_length, num_time_features)
        future_time_features = torch.randn(
            1, cfg.prediction_length, cfg.num_time_features, dtype=dtype
        )

        return {
            "past_values": past_values,
            "past_time_features": past_time_features,
            "past_observed_mask": past_observed_mask,
            "static_categorical_features": static_categorical_features,
            "future_time_features": future_time_features,
        }
