# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fashion-CLIP model loader implementation for JAX.
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
    """Available Fashion-CLIP model variants."""

    FASHION_CLIP = "Fashion_CLIP"


class ModelLoader(ForgeModel):
    """Fashion-CLIP model loader implementation for JAX."""

    _VARIANTS = {
        ModelVariant.FASHION_CLIP: ModelConfig(
            pretrained_model_name="patrickjohncyh/fashion-clip",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FASHION_CLIP

    sample_text = ["a blue striped t-shirt", "a red evening dress"]

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
            model="Fashion-CLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant.

        Args:
            dtype_override: Optional dtype to override the processor's default dtype.

        Returns:
            processor: The loaded processor instance
        """
        from transformers import CLIPProcessor

        processor_kwargs = {"do_rescale": False}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = CLIPProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Fashion-CLIP model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        from transformers import FlaxCLIPModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FlaxCLIPModel.from_pretrained(
            pretrained_model_name, from_pt=True, **model_kwargs
        )

        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Fashion-CLIP model.

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

        inputs = self._processor(
            text=self.sample_text,
            images=image,
            return_tensors="jax",
            padding=True,
        )

        return inputs

    def wrapper_model(self, f):
        """Wrapper for model forward method that extracts the appropriate output.

        Fashion-CLIP models output both image and text embeddings. This wrapper
        extracts the text model pooler output for compatibility with the testing
        framework.

        Args:
            f: The model forward function to wrap

        Returns:
            Wrapped function that extracts text model pooler output
        """

        def model(args, kwargs):
            out = f(*args, **kwargs)
            out = out.text_model_output.pooler_output
            return out

        return model
