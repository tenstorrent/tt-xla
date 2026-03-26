# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ViT feature extraction model loader implementation for JAX.
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
    """Available ViT feature extraction model variants."""

    BASE_PATCH16_224_IN_21K = "Base_Patch16_224_In_21K"


class ModelLoader(ForgeModel):
    """ViT feature extraction model loader implementation for JAX."""

    _VARIANTS = {
        ModelVariant.BASE_PATCH16_224_IN_21K: ModelConfig(
            pretrained_model_name="google/vit-base-patch16-224-in21k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_PATCH16_224_IN_21K

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._processor = None
        self._model_name = self._variant_config.pretrained_model_name

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
            model="ViT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
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

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = ViTImageProcessor.from_pretrained(
            self._model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ViT feature extraction model with the current variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded ViT model instance
        """
        from transformers import FlaxViTModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FlaxViTModel.from_pretrained(self._model_name, **model_kwargs)

        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ViT model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        from datasets import load_dataset

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = self._processor(image, return_tensors="jax")

        return inputs
