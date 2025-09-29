# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
RegNet model loader implementation for image classification.
"""

from typing import Optional, Any, Mapping
import jax

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
    """Available RegNet model variants."""

    REGNET_Y_040 = "y-040"
    REGNET_Y_160 = "y-160"
    REGNET_Y_320 = "y-320"


class ModelLoader(ForgeModel):
    """RegNet model loader implementation for image classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.REGNET_Y_040: ModelConfig(
            pretrained_model_name="facebook/regnet-y-040",
        ),
        ModelVariant.REGNET_Y_160: ModelConfig(
            pretrained_model_name="facebook/regnet-y-160",
        ),
        ModelVariant.REGNET_Y_320: ModelConfig(
            pretrained_model_name="facebook/regnet-y-320",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.REGNET_Y_040

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._processor = None

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
            model="regnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_processor(self):
        """Load the image processor for the current variant."""
        from transformers import AutoImageProcessor

        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        return self._processor

    def load_model(self, dtype_override=None):
        """Load and return the RegNet model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        from transformers import FlaxRegNetForImageClassification

        model_kwargs = {"from_pt": True}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        model = FlaxRegNetForImageClassification.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the RegNet model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        from datasets import load_dataset

        # Load the processor
        processor = self._load_processor()

        # Load a sample image from open-source cats-image dataset
        dataset = load_dataset("huggingface/cats-image", split="test")
        sample = dataset[0]

        # Process the image using the processor
        inputs = processor(images=sample["image"], return_tensors="jax")

        if dtype_override is not None:
            # Apply dtype override to all tensors in the inputs
            for key, value in inputs.items():
                if hasattr(value, "astype"):
                    inputs[key] = value.astype(dtype_override)

        return inputs
