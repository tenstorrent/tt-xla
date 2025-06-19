# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UNet model loader implementation
"""
import torch

from ...base import ForgeModel
from .src.unet import UNET


class ModelLoader(ForgeModel):
    """UNet model loader implementation."""

    # Shared configuration parameters
    input_shape = (3, 224, 224)

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the UNet model instance with default settings."""

        model = UNET(in_channels=3, out_channels=1)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Load and return sample inputs for the UNet model with default settings."""

        # Create a random input tensor with the correct shape, using default dtype
        inputs = torch.rand(1, *cls.input_shape)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
