# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DINOv2 model loader implementation for image classification.
"""

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
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available DINOv2 model variants."""

    BASE = "base"
    GIANT = "giant"
    LARGE = "large"


class ModelLoader(ForgeModel):
    """DINOv2 model loader implementation for image classification."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="facebook/dinov2-base",
        ),
        ModelVariant.GIANT: ModelConfig(
            pretrained_model_name="facebook/dinov2-giant",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="facebook/dinov2-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._processor = None
        self._model = self._variant_config.pretrained_model_name

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
            model="dinov2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_processor(self, dtype_override=None):
        """Load image processor for the current variant.
        Args:
            dtype_override: Optional dtype to override the processor's default dtype.
        Returns:
            processor: The loaded image processor instance
        """
        from transformers import AutoImageProcessor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = AutoImageProcessor.from_pretrained(
            self._model, **processor_kwargs
        )

        return self._processor

    def load_model(self, dtype_override=None):
        """Load the DINOv2 model with the current variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
        Returns:
            model: The loaded DINOv2 model instance
        """
        from transformers import FlaxDinov2ForImageClassification

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        model = FlaxDinov2ForImageClassification.from_pretrained(
            self._model, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the DINOv2 model with this instance's variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        from datasets import load_dataset

        # Ensure processor is initialized
        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Load dataset
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        # Process the image
        inputs = self._processor(image, return_tensors="jax")

        return inputs
