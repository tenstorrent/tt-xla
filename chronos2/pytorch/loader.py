# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chronos-2 model loader implementation for time series forecasting.
"""

from typing import Optional

import torch

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
    """Available Chronos-2 model variants."""

    CHRONOS_2 = "Chronos_2"
    CHRONOS_2_SMALL = "Chronos_2_Small"


class ModelLoader(ForgeModel):
    """Chronos-2 model loader implementation for time series forecasting."""

    _VARIANTS = {
        ModelVariant.CHRONOS_2: ModelConfig(
            pretrained_model_name="amazon/chronos-2",
        ),
        ModelVariant.CHRONOS_2_SMALL: ModelConfig(
            pretrained_model_name="autogluon/chronos-2-small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHRONOS_2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Chronos2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Chronos-2 model."""
        from chronos import Chronos2Model

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Chronos2Model.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series input for the Chronos-2 model.

        Returns a batch of synthetic time series context data.
        """
        # Chronos-2 expects context tensor of shape (batch_size, context_length)
        # Use 512 time steps as a reasonable context length
        context = torch.randn(1, 512)

        if dtype_override is not None:
            context = context.to(dtype_override)

        return {"context": context}
