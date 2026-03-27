# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CT FM Feature Extractor model loader implementation
"""

import torch
from typing import Optional

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
    """Available CT FM Feature Extractor model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """CT FM Feature Extractor model loader implementation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="project-lighter/ct_fm_feature_extractor",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CT FM Feature Extractor",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the CT FM Feature Extractor model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The loaded SegResEncoder model instance.
        """
        from lighter_zoo import SegResEncoder

        model = SegResEncoder.from_pretrained(self._model_name, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the CT FM Feature Extractor model.

        The model expects a 5D tensor of shape (batch, channels, depth, height, width)
        with 1 input channel, representing a 3D CT volume.

        Args:
            dtype_override: Optional torch.dtype to override the input tensor dtype.

        Returns:
            torch.Tensor: A sample 3D input tensor.
        """
        input_tensor = torch.randn(1, 1, 64, 64, 64)

        if dtype_override is not None:
            input_tensor = input_tensor.to(dtype_override)

        return input_tensor
