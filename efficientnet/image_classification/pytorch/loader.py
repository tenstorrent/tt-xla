# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EfficientNet model loader implementation for image classification via HuggingFace.
"""
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
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
    """Available EfficientNet model variants for image classification."""

    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    B5 = "B5"
    B6 = "B6"
    B7 = "B7"


class ModelLoader(ForgeModel):
    """EfficientNet model loader implementation for image classification tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.B0: ModelConfig(
            pretrained_model_name="google/efficientnet-b0",
        ),
        ModelVariant.B1: ModelConfig(
            pretrained_model_name="google/efficientnet-b1",
        ),
        ModelVariant.B2: ModelConfig(
            pretrained_model_name="google/efficientnet-b2",
        ),
        ModelVariant.B3: ModelConfig(
            pretrained_model_name="google/efficientnet-b3",
        ),
        ModelVariant.B4: ModelConfig(
            pretrained_model_name="google/efficientnet-b4",
        ),
        ModelVariant.B5: ModelConfig(
            pretrained_model_name="google/efficientnet-b5",
        ),
        ModelVariant.B6: ModelConfig(
            pretrained_model_name="google/efficientnet-b6",
        ),
        ModelVariant.B7: ModelConfig(
            pretrained_model_name="google/efficientnet-b7",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.B0

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
            model="EfficientNet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load image processor for the current variant.

        Returns:
            The loaded processor instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the EfficientNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The EfficientNet model instance for image classification.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the EfficientNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Load dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Process images
        inputs = self.processor(images=image, return_tensors="pt")

        # Handle batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Convert the input dtype to dtype_override if specified
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
