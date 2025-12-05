# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DenseUNet3d model loader implementation
"""
import torch
from typing import Optional
from .src.model import DenseUNet3d

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
from ...tools.utils import print_compiled_model_results


class ModelVariant(StrEnum):
    """Available Attention DenseUNet model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """Attention DenseUNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="dense_unet_3d",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant name string. If None, uses DEFAULT_VARIANT.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="dense_unet_3d",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the DenseUNet3d model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DenseUNet3d model instance.
        """
        # Load the DenseUNet3d model
        model = DenseUNet3d()

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for Attention DenseUNet model.

        Args:
            dtype_override: Optional torch.dtype to override the input's default dtype.
                           If not provided, inputs will use default dtype (typically float32).
            batch_size: Number of samples in the batch. Default is 1.

        Returns:
            torch.Tensor: Sample input tensor of shape (batch_size, 3, 256, 256)
        """
        # Create random input tensor: (batch_size, channels, height, width)
        input_tensor = torch.randn((1, 1, 32, 256, 256), dtype=torch.float32)

        if dtype_override is not None:
            input_tensor = input_tensor.to(dtype_override)

        return input_tensor

    def print_cls_results(self, compiled_model_out):
        """Print compiled model results.

        Args:
            compiled_model_out: Output from the compiled model
        """
        print_compiled_model_results(compiled_model_out)
