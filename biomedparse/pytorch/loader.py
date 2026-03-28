# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BiomedParse model loader implementation for biomedical image segmentation.

BiomedParse is a biomedical foundation model that jointly performs image
segmentation, object detection, and object recognition across 9 biomedical
imaging modalities (CT, MRI, X-Ray, Pathology, etc.) using text-guided prompts.

Reference: https://github.com/microsoft/BiomedParse
HuggingFace: https://huggingface.co/microsoft/BiomedParse
"""

import torch
from PIL import Image
from typing import Optional
from torchvision import transforms

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available BiomedParse model variants."""

    V1 = "v1"


class ModelLoader(ForgeModel):
    """BiomedParse model loader for biomedical image segmentation."""

    _VARIANTS = {
        ModelVariant.V1: ModelConfig(
            pretrained_model_name="microsoft/BiomedParse",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BiomedParse",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BiomedParse model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BiomedParse model instance.
        """
        from .src.model import build_biomedparse_model

        model = build_biomedparse_model()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the BiomedParse model.

        BiomedParse expects a 1024x1024 RGB image tensor normalized with
        ImageNet statistics.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape (batch_size, 3, 1024, 1024).
        """
        image_size = (1024, 1024)
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Load sample image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = transform(image).unsqueeze(0)

        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
