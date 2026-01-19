# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py
"""
Autoencoder Linear/Conv model loader implementation
"""
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from datasets import load_dataset
from typing import Optional
from dataclasses import dataclass

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from .src.linear_ae import LinearAE
from .src.conv_ae import ConvAE


@dataclass
class AEConfig(ModelConfig):
    source: ModelSource = ModelSource.CUSTOM


class ModelVariant(StrEnum):
    LINEAR = "linear"
    CONV = "conv"


class ModelLoader(ForgeModel):
    """Autoencoder model loader implementation (Linear and Conv variants)."""

    _VARIANTS = {
        ModelVariant.LINEAR: AEConfig(pretrained_model_name="linear"),
        ModelVariant.CONV: AEConfig(pretrained_model_name="conv"),
    }

    DEFAULT_VARIANT = ModelVariant.LINEAR

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="autoencoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Autoencoder model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Autoencoder model instance for image reconstruction.
        """
        if self._variant == ModelVariant.LINEAR:
            model = LinearAE()
        elif self._variant == ModelVariant.CONV:
            if ConvAE is None:
                raise ImportError(
                    "ConvAE is unavailable. Ensure test dependency module is accessible: test.models.pytorch.vision.autoencoder.model_utils.conv_autoencoder"
                )
            model = ConvAE()
        else:
            raise ValueError(f"Unsupported variant: {self._variant}")

        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Autoencoder model with default settings.

        Returns:
            torch.Tensor: Input tensor suitable for the Autoencoder model.
        """

        if self._variant == ModelVariant.LINEAR:
            transform = transforms.Compose(
                [
                    transforms.Resize((1, 784)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            dataset = load_dataset("mnist")
            sample = dataset["train"][0]["image"]
            sample_tensor = transform(sample).view(1, -1)
            batch_tensor = sample_tensor.repeat(batch_size, 1)

        elif self._variant == ModelVariant.CONV:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            dataset = load_dataset("mnist")
            sample = dataset["train"][0]["image"]
            sample_tensor = transform(sample).unsqueeze(0)
            batch_tensor = sample_tensor.repeat(batch_size, 1, 1, 1)
        else:
            raise ValueError(f"Unsupported variant: {self._variant}")

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor

    def post_processing(self, co_out, save_path):
        """Post-process the model outputs and save the reconstructed images.

        Args:
            co_out: Model output from a forward pass
            save_path: Path to save the reconstructed images
        """
        os.makedirs(save_path, exist_ok=True)

        if self._variant == ModelVariant.LINEAR:
            output_image = co_out[0].view(1, 28, 28).detach().numpy()
            reconstructed_image_path = f"{save_path}/reconstructed_image_linear.png"
            image_array = np.squeeze(output_image)
            image_array = (image_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array, mode="L")
            pil_image.save(reconstructed_image_path)
        elif self._variant == ModelVariant.CONV:
            output_image = co_out[0].detach().cpu().numpy()
            image_array = np.squeeze(output_image)
            image_array = (image_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array, mode="L")
            reconstructed_image_path = f"{save_path}/reconstructed_image_conv.png"
            pil_image.save(reconstructed_image_path)
