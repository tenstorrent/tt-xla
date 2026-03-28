# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MOMENT-1 model loader implementation for time series forecasting.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from momentfm import MOMENTPipeline

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
class MomentConfig(ModelConfig):
    seq_len: int = 512
    forecast_horizon: int = 96


class ModelVariant(StrEnum):
    BASE = "base"


class ModelLoader(ForgeModel):
    """MOMENT-1 model loader for time series forecasting.

    Loads the MOMENT-1 time series foundation model for
    zero-shot time series forecasting.
    """

    _VARIANTS = {
        ModelVariant.BASE: MomentConfig(
            pretrained_model_name="AutonLab/MOMENT-1-base",
            seq_len=512,
            forecast_horizon=96,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MOMENT-1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the MOMENT-1 model for time series forecasting.

        Returns:
            torch.nn.Module: The MOMENT-1 forecasting model instance.
        """
        cfg = self._variant_config

        model = MOMENTPipeline.from_pretrained(
            cfg.pretrained_model_name,
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": cfg.forecast_horizon,
            },
        )
        model.init()

        if dtype_override:
            model = model.to(dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            torch.Tensor: Input tensor of shape (batch_size, n_channels, seq_len).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        inputs = torch.randn(1, 1, cfg.seq_len, dtype=dtype)

        return inputs
