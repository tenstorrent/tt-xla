# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py
"""
Autoencoder Linear model loader implementation
"""
import torch
import torchvision.transforms as transforms
from datasets import load_dataset

from ...base import ForgeModel
from .src.linear_ae import LinearAE


class ModelLoader(ForgeModel):
    """Autoencoder linear model loader implementation."""

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Autoencoder Linear model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Autoencoder Linear model instance.
        """
        model = LinearAE()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Autoencoder Linear model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: A batch of input tensors that can be fed to the model.
        """

        # Define transform to normalize data
        transform = transforms.Compose(
            [
                transforms.Resize((1, 784)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Load sample from MNIST dataset
        dataset = load_dataset("mnist")
        sample = dataset["train"][0]["image"]
        batch_tensor = torch.stack([transform(sample)] * batch_size)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
