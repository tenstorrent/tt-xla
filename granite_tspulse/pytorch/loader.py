# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite TSPulse model loader implementation for time series anomaly detection.
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
class GraniteTSPulseConfig(ModelConfig):
    context_length: int = 512


class ModelVariant(StrEnum):
    R1 = "r1"


class ModelLoader(ForgeModel):
    """Granite TSPulse model loader for time series anomaly detection.

    Loads the IBM Granite TSPulse R1 model for zero-shot
    time series anomaly detection using dual-space masked reconstruction.
    """

    _VARIANTS = {
        ModelVariant.R1: GraniteTSPulseConfig(
            pretrained_model_name="ibm-granite/granite-timeseries-tspulse-r1",
            context_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.R1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Granite-TSPulse",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Granite TSPulse model for time series anomaly detection.

        Returns:
            torch.nn.Module: The TSPulse reconstruction model instance.
        """
        cfg = self._variant_config

        from tsfm_public.models.tspulse import TSPulseForReconstruction

        model = TSPulseForReconstruction.from_pretrained(cfg.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'past_values' tensor of shape
                  (batch, context_length, num_channels).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        # TSPulse expects (batch, context_length, num_input_channels)
        past_values = torch.randn(1, cfg.context_length, 1, dtype=dtype)

        return {"past_values": past_values}
