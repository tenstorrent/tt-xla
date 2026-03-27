# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chronos-Bolt model loader implementation for time series forecasting.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from chronos import ChronosBoltPipeline

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
class ChronosBoltConfig(ModelConfig):
    context_length: int = 512
    prediction_length: int = 64


class ModelVariant(StrEnum):
    TINY = "tiny"
    MINI = "mini"
    SMALL = "small"
    BASE = "base"
    AMAZON_SMALL = "amazon-small"


class ModelLoader(ForgeModel):
    """Chronos-Bolt model loader for time series forecasting."""

    _VARIANTS = {
        ModelVariant.TINY: ChronosBoltConfig(
            pretrained_model_name="autogluon/chronos-bolt-tiny",
            context_length=512,
            prediction_length=64,
        ),
        ModelVariant.MINI: ChronosBoltConfig(
            pretrained_model_name="amazon/chronos-bolt-mini",
            context_length=512,
            prediction_length=64,
        ),
        ModelVariant.SMALL: ChronosBoltConfig(
            pretrained_model_name="autogluon/chronos-bolt-small",
            context_length=512,
            prediction_length=64,
        ),
        ModelVariant.BASE: ChronosBoltConfig(
            pretrained_model_name="amazon/chronos-bolt-base",
            context_length=512,
            prediction_length=64,
        ),
        ModelVariant.AMAZON_SMALL: ChronosBoltConfig(
            pretrained_model_name="amazon/chronos-bolt-small",
            context_length=512,
            prediction_length=64,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Chronos-Bolt",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Chronos-Bolt model for time series forecasting.

        Returns:
            torch.nn.Module: The ChronosBoltModelForForecasting instance.
        """
        cfg = self._variant_config

        pipeline = ChronosBoltPipeline.from_pretrained(
            cfg.pretrained_model_name,
            device_map="cpu",
            dtype=dtype_override or torch.float32,
        )

        model = pipeline.model
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'context' tensor of shape (batch, context_length).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        # Generate a synthetic time series as sample input
        torch.manual_seed(42)
        context = torch.randn(1, cfg.context_length, dtype=dtype)

        return {"context": context}
