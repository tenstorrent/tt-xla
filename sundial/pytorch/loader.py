# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sundial model loader implementation for time series forecasting.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from transformers import AutoModelForCausalLM

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
class SundialConfig(ModelConfig):
    context_length: int = 2880
    prediction_length: int = 96


class ModelVariant(StrEnum):
    BASE_128M = "base-128m"


class ModelLoader(ForgeModel):
    """Sundial model loader for time series forecasting.

    Loads the Sundial causal transformer model for zero-shot
    time series point and probabilistic forecasting.
    """

    _VARIANTS = {
        ModelVariant.BASE_128M: SundialConfig(
            pretrained_model_name="thuml/sundial-base-128m",
            context_length=2880,
            prediction_length=96,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_128M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Sundial",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Sundial model for time series forecasting.

        Returns:
            torch.nn.Module: The Sundial causal LM model instance.
        """
        cfg = self._variant_config

        model = AutoModelForCausalLM.from_pretrained(
            cfg.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=dtype_override or torch.float32,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'input_ids' float tensor of shape
                  (batch_size, context_length).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        input_ids = torch.randn(1, cfg.context_length, dtype=dtype)

        return {"input_ids": input_ids}
