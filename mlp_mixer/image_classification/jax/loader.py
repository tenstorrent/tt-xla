# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MLP Mixer model loader implementation for image classification.
"""

from typing import Optional
import inspect
import jax
import ml_collections
import numpy as np
import flax.traverse_util
import fsspec

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
from .src.model_implementation import MlpMixer


class ModelVariant(StrEnum):
    """Available MLP Mixer model variants."""

    BASE_16 = "base_16"


class ModelLoader(ForgeModel):
    """MLP Mixer model loader implementation for image classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_16: ModelConfig(
            pretrained_model_name="mixer-b16",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_16

    # Hyperparameters for Mixer-B/16
    _MIXER_B16_CONFIG = {
        "patch_size": 16,
        "num_classes": 21843,
        "num_blocks": 12,
        "hidden_dim": 768,
        "tokens_mlp_dim": 384,
        "channels_mlp_dim": 3072,
    }

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
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="mlpmixer",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the MLP Mixer model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            model: The loaded model instance
        """

        if self._variant == ModelVariant.BASE_16:
            config = self._MIXER_B16_CONFIG
            patch = ml_collections.ConfigDict(
                {"size": (config["patch_size"], config["patch_size"])}
            )

            return MlpMixer(
                patches=patch,
                num_classes=config["num_classes"],
                num_blocks=config["num_blocks"],
                hidden_dim=config["hidden_dim"],
                tokens_mlp_dim=config["tokens_mlp_dim"],
                channels_mlp_dim=config["channels_mlp_dim"],
            )
        else:
            raise ValueError(f"Unsupported variant: {self._variant}")

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MLP Mixer model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        from datasets import load_dataset
        from PIL import Image

        # Load a sample image from a publicly available dataset
        dataset = load_dataset("cifar10", split="test")
        # Get the first image from the dataset
        image = dataset[0]["img"]

        # Resize to 224x224 for MLP Mixer
        # Using Lanczos resampling for high-quality image resizing with better edge preservation
        # and anti-aliasing, especially important when upscaling from 32x32 to 224x224
        image = image.resize((224, 224), Image.LANCZOS)

        # Convert PIL image to numpy array
        img_array = np.array(image)
        # Convert to float and normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        # Add batch dimension
        img_array = img_array[np.newaxis, ...]  # (1, 224, 224, 3)

        # Convert to JAX array
        return jax.numpy.array(img_array)

    def load_parameters(self, dtype_override=None):
        """Load and return model parameters.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            PyTree: Model parameters loaded from pretrained weights
        """
        # Download and load pretrained weights
        # TODO: update this to download weights from S3
        link = "https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz"
        with fsspec.open("filecache::" + link, cache_storage="/tmp/files/") as f:
            weights = np.load(f, encoding="bytes")
            state_dict = {k: v for k, v in weights.items()}
            return {"params": flax.traverse_util.unflatten_dict(state_dict, sep="/")}

    def get_input_parameters(self, dtype_override=None):
        """Get input parameters for the model.

        This method provides compatibility with DynamicJaxModelTester which expects
        a get_input_parameters method.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            PyTree: Model parameters loaded from pretrained weights
        """
        # Delegate to load_parameters which already implements the logic
        return self.load_parameters(dtype_override=dtype_override)

    def get_forward_method_kwargs(self, run_mode=None):
        """Get keyword arguments for the model's forward method.

        This method provides compatibility with DynamicJaxModelTester by returning
        the appropriate kwargs based on what the model's __call__ method accepts.

        Args:
            run_mode: Optional RunMode. If not provided, defaults to inference behavior.

        Returns:
            dict: Keyword arguments for the model's forward method
        """
        # Load the model to inspect its signature
        model = self.load_model()

        # Get the signature of the model's __call__ method
        try:
            sig = inspect.signature(model.__call__)
            params = sig.parameters
        except (AttributeError, ValueError):
            # If we can't inspect the signature, return empty kwargs
            return {}

        kwargs = {}

        # Check if the model accepts a 'train' parameter
        if "train" in params:
            # Determine if we're in training mode
            # RunMode.TRAINING has string representation 'training'
            is_training = str(run_mode).lower() == "training" if run_mode else False
            kwargs["train"] = is_training

        # MlpMixer doesn't have a train parameter, so this will return empty dict
        return kwargs

    def get_forward_method_name(self):
        """Get the name of the forward method for the model.

        Returns:
            str: Name of the forward method (typically 'apply' for Flax models)
        """
        return "apply"
