# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Resnet model loader implementation for question answering
"""
import torch

import requests
from PIL import Image
from transformers import ResNetForImageClassification

from ...base import ForgeModel


class ModelLoader(ForgeModel):
    # Shared configuration parameters
    model_name = "microsoft/resnet-50"
    input_shape = (3, 224, 224)

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load a Resnet model from Hugging Face."""
        model = ResNetForImageClassification.from_pretrained(
            cls.model_name, return_dict=False
        )

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Generate sample inputs for Resnet models."""

        # Create a random input tensor with the correct shape, using default dtype
        inputs = torch.rand(1, *cls.input_shape)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
