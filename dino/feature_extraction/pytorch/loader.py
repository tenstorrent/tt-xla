# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DINO ViT model loader implementation for feature extraction (PyTorch).
"""

import torch
from datasets import load_dataset
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DINO ViT feature extraction model variants."""

    SMALL_8 = "Small_8"


class ModelLoader(ForgeModel):
    """DINO ViT model loader implementation for feature extraction (PyTorch)."""

    _VARIANTS = {
        ModelVariant.SMALL_8: ModelConfig(
            pretrained_model_name="facebook/dino-vits8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_8

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DINO ViT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load image processor for the current variant.

        Returns:
            The loaded processor instance
        """
        from transformers import ViTImageProcessor

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DINO ViT model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The DINO ViT model instance for feature extraction.
        """
        from transformers import ViTModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ViTModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the DINO ViT model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.
            image: Optional input image. If None, loads from HuggingFace datasets.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        if self.processor is None:
            self._load_processor()

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
