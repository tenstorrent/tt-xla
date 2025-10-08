# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlexNet model loader implementation for image classification.
"""

from typing import Optional
import jax.numpy as jnp

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
from .src import AlexNetModel


class ModelVariant(StrEnum):
    """Available AlexNet model variants."""

    CUSTOM = "custom"
    CUSTOM_1X2 = "custom_1x2"
    CUSTOM_1X4 = "custom_1x4"
    CUSTOM_1X8 = "custom_1x8"


class ModelLoader(ForgeModel):
    """AlexNet model loader implementation for image classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.CUSTOM: ModelConfig(
            pretrained_model_name="custom",
        ),
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="custom_1x2",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="custom_1x4",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="custom_1x8",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.CUSTOM

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to get info for.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="alexnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the AlexNet model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        # Apply dtype override if specified
        if dtype_override is not None:
            model = AlexNetModel(param_dtype=dtype_override)
        else:
            model = AlexNetModel()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the AlexNet model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        from datasets import load_dataset
        import numpy as np

        # Load a sample image from open-source cats-image dataset
        dataset = load_dataset("huggingface/cats-image", split="test")
        sample = dataset[0]

        # Resize to 224x224 (AlexNet input size)
        image = sample["image"].resize((224, 224))
        image = np.array(image)

        # Normalize to [-128, 127] range as per original paper
        image = image.astype(np.float32)
        image = image - 128.0
        image = np.clip(image, -128, 127)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Convert to JAX array
        inputs = jnp.array(image)

        # Apply dtype override if specified
        if dtype_override is not None:
            inputs = inputs.astype(dtype_override)

        return inputs
