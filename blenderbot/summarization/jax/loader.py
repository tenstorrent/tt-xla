# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Blenderbot model loader implementation"""

from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available Blenderbot model variants."""

    BLENDERBOT_3B = "3B"
    BLENDERBOT_SMALL_90M = "small-90M"
    BLENDERBOT_1B_DISTILL = "1B-distill"
    BLENDERBOT_400M_DISTILL = "400M-distill"


class ModelLoader(ForgeModel):
    """Blenderbot model loader implementation for summarization."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BLENDERBOT_3B: LLMModelConfig(
            pretrained_model_name="facebook/blenderbot-3B",
        ),
        ModelVariant.BLENDERBOT_SMALL_90M: LLMModelConfig(
            pretrained_model_name="facebook/blenderbot_small-90M",
        ),
        ModelVariant.BLENDERBOT_1B_DISTILL: LLMModelConfig(
            pretrained_model_name="facebook/blenderbot-1B-distill",
        ),
        ModelVariant.BLENDERBOT_400M_DISTILL: LLMModelConfig(
            pretrained_model_name="facebook/blenderbot-400M-distill",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BLENDERBOT_3B

    # Shared configuration parameters
    sample_text = """summarize: Researchers have extensively studied the benefits of having pets,
                    particularly dogs, on human health and well-being. Findings suggest that pet ownership
                    can lead to improved mental health, reduced stress levels, and even physical health benefits
                    such as lower blood pressure and increased physical activity levels."""

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
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
        return ModelInfo(
            model="blenderbot",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance.
        """
        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        from transformers import AutoTokenizer

        # Load the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Blenderbot model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            The loaded model instance.
        """

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        from transformers import FlaxBlenderbotForConditionalGeneration

        model = FlaxBlenderbotForConditionalGeneration.from_pretrained(
            self._model_name, from_pt=True, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Blenderbot model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs
        inputs = self._tokenizer(self.sample_text, return_tensors="jax")

        return inputs
