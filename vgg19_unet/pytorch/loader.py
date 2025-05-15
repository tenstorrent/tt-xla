# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VGG19-UNet model loader implementation
"""
import torch

from ...base import ForgeModel
from .src.vgg19_unet import VGG19UNet


class ModelLoader(ForgeModel):
    """VGG19-UNet model loader implementation."""

    # Shared configuration parameters
    input_shape = (3, 512, 512)
    out_channels = 1

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the VGG19-UNet model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The VGG19-UNet model instance.
        """
        # Load model with defaults
        model = VGG19UNet(input_shape=cls.input_shape, out_channels=cls.out_channels)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Load and return sample inputs for the VGG19-UNet model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Create a random input tensor with the correct shape, using default dtype
        inputs = torch.rand(1, *cls.input_shape)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
