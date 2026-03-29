# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UperNet for Semantic Segmentation model loader implementation
"""

import torch
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from datasets import load_dataset
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available UperNet model variants."""

    SWIN_TINY = "Swin_Tiny"
    SWIN_SMALL = "Swin_Small"
    SWIN_BASE = "Swin_Base"
    SWIN_LARGE = "Swin_Large"


class ModelLoader(ForgeModel):
    """UperNet model loader implementation for semantic segmentation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.SWIN_TINY: ModelConfig(
            pretrained_model_name="openmmlab/upernet-swin-tiny",
        ),
        ModelVariant.SWIN_SMALL: ModelConfig(
            pretrained_model_name="openmmlab/upernet-swin-small",
        ),
        ModelVariant.SWIN_BASE: ModelConfig(
            pretrained_model_name="openmmlab/upernet-swin-base",
        ),
        ModelVariant.SWIN_LARGE: ModelConfig(
            pretrained_model_name="openmmlab/upernet-swin-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SWIN_SMALL

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
            model="UperNet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UperNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The UperNet model instance for semantic segmentation.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self.processor is None:
            self._load_processor()

        # Load pre-trained model from HuggingFace
        model = UperNetForSemanticSegmentation.from_pretrained(
            pretrained_model_name, **kwargs
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UperNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (pixel_values) that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]
        inputs = self.processor(images=image, return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
