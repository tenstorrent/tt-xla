# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CleanTS model loader implementation for time series forecasting.
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
class CleanTSConfig(ModelConfig):
    context_length: int = 512
    patch_size: int = 32


class ModelVariant(StrEnum):
    BASE_65M = "base-65m"


class ModelLoader(ForgeModel):
    """CleanTS model loader for time series forecasting.

    Loads the EINK CleanTS encoder-only transformer model for
    zero-shot time series quantile forecasting.
    """

    _VARIANTS = {
        ModelVariant.BASE_65M: CleanTSConfig(
            pretrained_model_name="EINK/CleanTS-65M",
            context_length=512,
            patch_size=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_65M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CleanTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the CleanTS model for time series forecasting.

        Returns:
            torch.nn.Module: The CleanTS model instance.
        """
        from .src.model import CleanTS

        cfg = self._variant_config

        model = CleanTS.from_pretrained(cfg.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'past_values' tensor of shape
                  (batch_size, context_length).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        past_values = torch.randn(1, cfg.context_length, dtype=dtype)

        return {"past_values": past_values}
