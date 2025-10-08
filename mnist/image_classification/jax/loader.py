# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MNIST model loader implementation for image classification.
"""

from typing import Optional, Sequence
import jax
import numpy as np

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
from .mlp.model_implementation import MNISTMLPModel
from .cnn_batchnorm.model_implementation import MNISTCNNBatchNormModel
from .cnn_dropout.model_implementation import MNISTCNNDropoutModel


class ModelVariant(StrEnum):
    """Available MNIST model architectures."""

    MLP_CUSTOM = "mlp_custom"
    MLP_CUSTOM_1X2 = "mlp_custom_1x2"
    MLP_CUSTOM_1X4 = "mlp_custom_1x4"
    MLP_CUSTOM_1X8 = "mlp_custom_1x8"
    CNN_BATCHNORM = "cnn_batchnorm"
    CNN_DROPOUT = "cnn_dropout"


class ModelLoader(ForgeModel):
    """MNIST model loader implementation for image classification."""

    # Dictionary of available model configurations
    _VARIANTS = {
        ModelVariant.MLP_CUSTOM: ModelConfig(
            pretrained_model_name="mnist_mlp_custom",
        ),
        ModelVariant.MLP_CUSTOM_1X2: ModelConfig(
            pretrained_model_name="mnist_mlp_custom_1x2",
        ),
        ModelVariant.MLP_CUSTOM_1X4: ModelConfig(
            pretrained_model_name="mnist_mlp_custom_1x4",
        ),
        ModelVariant.MLP_CUSTOM_1X8: ModelConfig(
            pretrained_model_name="mnist_mlp_custom_1x8",
        ),
        ModelVariant.CNN_BATCHNORM: ModelConfig(
            pretrained_model_name="mnist_cnn_batchnorm_custom",
        ),
        ModelVariant.CNN_DROPOUT: ModelConfig(
            pretrained_model_name="mnist_cnn_dropout_custom",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MLP_CUSTOM

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        hidden_sizes: Sequence[int] = (256, 128, 64),
    ):
        """Initialize ModelLoader with specified variant and configuration.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            hidden_sizes: Hidden layer sizes for MLP architecture.
        """
        super().__init__(variant)
        self._hidden_sizes = tuple(hidden_sizes)

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
            model="mnist",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the MNIST model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            model: The loaded model instance
        """

        if self._variant in [
            ModelVariant.MLP_CUSTOM,
            ModelVariant.MLP_CUSTOM_1X2,
            ModelVariant.MLP_CUSTOM_1X4,
            ModelVariant.MLP_CUSTOM_1X8,
        ]:
            return MNISTMLPModel(self._hidden_sizes)
        elif self._variant == ModelVariant.CNN_BATCHNORM:
            return MNISTCNNBatchNormModel()
        elif self._variant == ModelVariant.CNN_DROPOUT:
            return MNISTCNNDropoutModel()
        else:
            raise ValueError(f"Unsupported variant: {self._variant}")

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MNIST model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        from datasets import load_dataset

        # Load MNIST dataset from Hugging Face
        dataset = load_dataset("mnist", split="test")
        # Get the first image from the dataset
        image = dataset[0]["image"]

        # Convert PIL image to numpy array and add batch dimension
        img_array = np.array(image)
        # Convert to float and normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        # Add channel dimension (grayscale) and batch dimension
        img_array = img_array[np.newaxis, :, :, np.newaxis]  # (1, 28, 28, 1)

        # Convert to JAX array
        return jax.numpy.array(img_array)

    def load_parameters(self, dtype_override=None):
        """Load and return model parameters.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            PyTree: Model parameters initialized with random weights
        """
        model = self.load_model(dtype_override)
        inputs = self.load_inputs(dtype_override)

        # Handle different model signatures
        if self._variant in [
            ModelVariant.MLP_CUSTOM,
            ModelVariant.MLP_CUSTOM_1X2,
            ModelVariant.MLP_CUSTOM_1X4,
            ModelVariant.MLP_CUSTOM_1X8,
        ]:
            return model.init(jax.random.PRNGKey(42), inputs)
        else:
            # CNN models require train argument
            return model.init(jax.random.PRNGKey(42), inputs, train=False)
