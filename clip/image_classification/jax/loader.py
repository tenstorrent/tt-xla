# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CLIP model loader implementation for JAX.
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
    """Available CLIP model variants."""

    BASE_PATCH16 = "base_patch16"
    BASE_PATCH32 = "base_patch32"
    LARGE_PATCH14 = "large_patch14"
    LARGE_PATCH14_336 = "large_patch14_336"


class ModelLoader(ForgeModel):
    """CLIP model loader implementation for JAX."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_PATCH16: ModelConfig(
            pretrained_model_name="openai/clip-vit-base-patch16",
        ),
        ModelVariant.BASE_PATCH32: ModelConfig(
            pretrained_model_name="openai/clip-vit-base-patch32",
        ),
        ModelVariant.LARGE_PATCH14: ModelConfig(
            pretrained_model_name="openai/clip-vit-large-patch14",
        ),
        ModelVariant.LARGE_PATCH14_336: ModelConfig(
            pretrained_model_name="openai/clip-vit-large-patch14-336",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_PATCH32

    sample_text = ["a photo of a cat", "a photo of a dog"]

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
        # Use the provided variant or fall back to default
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="clip",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_CAPT,
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
        from transformers import CLIPProcessor

        # Initialize processor with dtype_override if provided

        processor_kwargs = {"do_rescale": False}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        # Load the processor
        self._processor = CLIPProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, dtype_override=None):
        """Load and return the CLIP model instance for this instance's variant.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
        Returns:
            model: The loaded model instance
        """

        from transformers import FlaxCLIPModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self._processor is None:
            self._load_processor(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Check if we need to load from PyTorch weights
        from_pt = pretrained_model_name == "openai/clip-vit-large-patch14-336"

        # Load the model
        model = FlaxCLIPModel.from_pretrained(
            pretrained_model_name, from_pt=from_pt, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the CLIP model with this instance's variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        from datasets import load_dataset

        # Ensure processor is initialized
        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Load a sample image from Hugging Face cats-image dataset

        dataset = load_dataset("huggingface/cats-image", split="test")
        # Get the first image from the dataset
        image = dataset[0]["image"]

        # Process the inputs
        inputs = self._processor(
            text=self.sample_text,
            images=image,
            return_tensors="jax",
        )

        return inputs

    def wrapper_model(self, f):
        """Wrapper for model forward method that extracts the appropriate output.

        CLIP models output both image and text embeddings. This wrapper extracts
        the text model pooler output for compatibility with the testing framework.

        Args:
            f: The model forward function to wrap

        Returns:
            Wrapped function that extracts text model pooler output
        """

        def model(args, kwargs):
            out = f(*args, **kwargs)
            # Extract text model pooler output from the CLIP model output
            out = out.text_model_output.pooler_output
            return out

        return model
