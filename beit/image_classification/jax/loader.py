# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
BEiT model loader implementation for image classification.
"""

import jax
from transformers import BeitImageProcessor, FlaxBeitForImageClassification
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
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available BEiT model variants."""

    BASE = "base"
    LARGE = "large"


class ModelLoader(ForgeModel):
    """BEiT model loader implementation for image classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="microsoft/beit-base-patch16-224",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="microsoft/beit-large-patch16-224",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant=None):
        """Implementation method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="beit",
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

        # Initialize processor with dtype_override if provided

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        # Load the processor
        self._processor = BeitImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, dtype_override=None):
        """Load and return the BEiT model instance for this instance's variant.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
        Returns:
            model: The loaded model instance
        """

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self._processor is None:
            self._load_processor(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        model = FlaxBeitForImageClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BEiT model with this instance's variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        # Ensure processor is initialized
        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Load dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Process the image
        inputs = self._processor(image, return_tensors="jax")

        return inputs
