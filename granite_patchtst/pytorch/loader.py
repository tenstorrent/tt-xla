# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite PatchTST model loader implementation for time series forecasting.
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


@dataclass
class GranitePatchTSTConfig(ModelConfig):
    context_length: int = 512
    prediction_length: int = 96
    num_input_channels: int = 7


class ModelVariant(StrEnum):
    PATCHTST = "patchtst"


class ModelLoader(ForgeModel):
    """Granite PatchTST model loader for time series forecasting.

    Loads IBM Granite PatchTST models for multivariate time series forecasting.
    The model segments time series into patches as input tokens to a Transformer encoder.
    """

    _VARIANTS = {
        ModelVariant.PATCHTST: GranitePatchTSTConfig(
            pretrained_model_name="ibm-granite/granite-timeseries-patchtst",
            context_length=512,
            prediction_length=96,
            num_input_channels=7,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PATCHTST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Granite-PatchTST",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Granite PatchTST model for time series forecasting.

        Returns:
            torch.nn.Module: The PatchTST model instance.
        """
        from transformers import PatchTSTForPrediction

        cfg = self._variant_config

        model = PatchTSTForPrediction.from_pretrained(cfg.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'past_values' tensor of shape
                  (batch, context_length, num_input_channels).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        # PatchTST expects (batch, context_length, num_input_channels)
        past_values = torch.randn(
            1, cfg.context_length, cfg.num_input_channels, dtype=dtype
        )

        return {"past_values": past_values}
