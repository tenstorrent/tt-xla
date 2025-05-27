# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VGG19-UNet model loader implementation
"""
import torch

from ...base import ForgeModel
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


class ModelLoader(ForgeModel):
    """Efficientnet model loader implementation."""

    # Shared configuration parameters
    input_shape = (3, 224, 224)

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Efficientnet model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Efficientnet model instance.
        """

        WeightsEnum.get_state_dict = get_state_dict
        # Load model with defaults
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Load and return sample inputs for the Efficientnet model with default settings.

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
