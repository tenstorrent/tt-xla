# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VIT model loader implementation for JAX.
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
    """Available VIT model variants."""

    BASE_PATCH16_224 = "base_patch16_224"
    BASE_PATCH16_384 = "base_patch16_384"
    BASE_PATCH32_224_IN_21K = "base_patch32_224_in_21k"
    BASE_PATCH32_384 = "base_patch32_384"
    HUGE_PATCH14_224_IN_21K = "huge_patch14_224_in_21k"
    LARGE_PATCH16_224 = "large_patch16_224"
    LARGE_PATCH16_384 = "large_patch16_384"
    LARGE_PATCH32_224_IN_21K = "large_patch32_224_in_21k"
    LARGE_PATCH32_384 = "large_patch32_384"


class ModelLoader(ForgeModel):
    """VIT model loader implementation for JAX."""

    _VARIANTS = {
        ModelVariant.BASE_PATCH16_224: ModelConfig(
            pretrained_model_name="google/vit-base-patch16-224",
        ),
        ModelVariant.BASE_PATCH16_384: ModelConfig(
            pretrained_model_name="google/vit-base-patch16-384",
        ),
        ModelVariant.BASE_PATCH32_224_IN_21K: ModelConfig(
            pretrained_model_name="google/vit-base-patch32-224-in21k",
        ),
        ModelVariant.BASE_PATCH32_384: ModelConfig(
            pretrained_model_name="google/vit-base-patch32-384",
        ),
        ModelVariant.HUGE_PATCH14_224_IN_21K: ModelConfig(
            pretrained_model_name="google/vit-huge-patch14-224-in21k",
        ),
        ModelVariant.LARGE_PATCH16_224: ModelConfig(
            pretrained_model_name="google/vit-large-patch16-224",
        ),
        ModelVariant.LARGE_PATCH16_384: ModelConfig(
            pretrained_model_name="google/vit-large-patch16-384",
        ),
        ModelVariant.LARGE_PATCH32_224_IN_21K: ModelConfig(
            pretrained_model_name="google/vit-large-patch32-224-in21k",
        ),
        ModelVariant.LARGE_PATCH32_384: ModelConfig(
            pretrained_model_name="google/vit-large-patch32-384",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_PATCH16_224

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
            model="vit",
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
        from transformers import ViTImageProcessor

        # Initialize processor with dtype_override if provided
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = ViTImageProcessor.from_pretrained(
            self._model, **processor_kwargs
        )

        return self._processor

    def load_model(self, dtype_override=None):
        """Load the VIT model with the current variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
        Returns:
            model: The loaded VIT model instance
        """
        from transformers import FlaxViTForImageClassification

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        model = FlaxViTForImageClassification.from_pretrained(
            self._model, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the VIT model with this instance's variant settings.
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
