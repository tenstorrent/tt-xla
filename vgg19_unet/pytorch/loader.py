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
    torch_dtype = torch.bfloat16

    @classmethod
    def load_model(cls):
        """Load and return the VGG19-UNet model instance with default settings.

        Returns:
            torch.nn.Module: The VGG19-UNet model instance.
        """
        # Load model with defaults
        model = VGG19UNet(input_shape=cls.input_shape, out_channels=cls.out_channels)

        return model.to(cls.torch_dtype)

    @classmethod
    def load_inputs(cls):
        """Load and return sample inputs for the VGG19-UNet model with default settings.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Create a random input tensor with the correct shape
        inputs = torch.rand(1, *cls.input_shape, dtype=cls.torch_dtype)

        return inputs
