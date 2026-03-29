# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PatchTST model loader implementation for time series forecasting.
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
class PatchTSTConfig(ModelConfig):
    context_length: int = 512
    prediction_length: int = 96
    num_input_channels: int = 7


class ModelVariant(StrEnum):
    TEST = "test"


class ModelLoader(ForgeModel):
    """PatchTST model loader for time series forecasting.

    Loads IBM Research PatchTST models for multivariate time series forecasting.
    """

    _VARIANTS = {
        ModelVariant.TEST: PatchTSTConfig(
            pretrained_model_name="ibm-research/test-patchtst",
            context_length=512,
            prediction_length=96,
            num_input_channels=7,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PatchTST",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the PatchTST model for time series forecasting.

        Returns:
            torch.nn.Module: The PatchTSTForPrediction model instance.
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
        past_values = torch.randn(
            1, cfg.context_length, cfg.num_input_channels, dtype=dtype
        )

        return {"past_values": past_values}
