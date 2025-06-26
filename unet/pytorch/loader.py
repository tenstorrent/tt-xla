# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UNet model loader implementation
"""
import torch

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from .src.unet import UNET


class ModelLoader(ForgeModel):
    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="unet",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    """UNet model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.input_shape = (3, 224, 224)

    def load_model(self, dtype_override=None):
        """Load and return the UNet model instance with default settings."""

        model = UNET(in_channels=3, out_channels=1)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the UNet model with default settings."""

        # Create a random input tensor with the correct shape, using default dtype
        inputs = torch.rand(1, *self.input_shape)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
