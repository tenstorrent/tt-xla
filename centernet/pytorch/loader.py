# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CenterNet model loader implementation
"""
import torch
import onnx
from PIL import Image
from torchvision import transforms
import numpy as np

from ...base import ForgeModel
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """CenterNet model loader implementation."""

    @classmethod
    def load_model(cls):
        """Load and return the CenterNet model instance with default settings.

        Returns:
            Onnx model: The CenterNet model instance.
        """
        # Load model with defaults
        file = get_file("test_files/onnx/centernet/centernet_resnet18.onnx")
        model = onnx.load(file)

        return model

    @classmethod
    def load_inputs(cls):
        """Load and return sample inputs for the CenterNet model with default settings.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Create a random input tensor with the correct shape, using default dtype
        image = (
            Image.open(
                get_file(
                    "test_files/onnx/centernet/inputs/17790319373_bd19b24cfc_k.jpg"
                )
            )
            .convert("RGB")
            .resize((512, 512))
        )
        m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=m, std=s),
            ]
        )
        input_tensor = preprocess(image)
        inputs = input_tensor.unsqueeze(0)

        return inputs
