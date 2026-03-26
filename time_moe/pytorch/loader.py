# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TimeMoE model loader implementation for time series forecasting.
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
class TimeMoEConfig(ModelConfig):
    context_length: int = 512
    prediction_length: int = 96


class ModelVariant(StrEnum):
    BASE_200M = "base-200m"


class ModelLoader(ForgeModel):
    """TimeMoE model loader for time series forecasting.

    Loads the TimeMoE Mixture of Experts causal model for
    zero-shot time series forecasting.
    """

    _VARIANTS = {
        ModelVariant.BASE_200M: TimeMoEConfig(
            pretrained_model_name="Maple728/TimeMoE-200M",
            context_length=512,
            prediction_length=96,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_200M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TimeMoE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the TimeMoE model for time series forecasting.

        Returns:
            torch.nn.Module: The TimeMoE causal LM model instance.
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
            torch.Tensor: Input tensor of shape (batch_size, context_length).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        inputs = torch.randn(1, cfg.context_length, dtype=dtype)

        return inputs
