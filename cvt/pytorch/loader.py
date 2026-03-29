# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CvT model loader implementation for image classification
"""
import torch
from transformers import AutoImageProcessor, CvtForImageClassification
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
    """Available CvT model variants."""

    CVT_13 = "cvt-13"


class ModelLoader(ForgeModel):
    """CvT model loader implementation for image classification tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.CVT_13: ModelConfig(
            pretrained_model_name="microsoft/cvt-13",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.CVT_13

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
        return ModelInfo(
            model="CvT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """

        # Initialize processor
        self.processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CvT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The CvT model instance for image classification.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self.processor is None:
            self._load_processor()

        # Load pre-trained model from HuggingFace
        model = CvtForImageClassification.from_pretrained(
            pretrained_model_name, **kwargs
        )

        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the CvT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Load dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Process the image
        inputs = self.processor(images=image, return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # Add batch dimension if batch_size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
