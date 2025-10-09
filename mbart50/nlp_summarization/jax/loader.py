# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MBart50 model loader implementation for summarization.
"""


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
    """Available MBart50 model variants."""

    LARGE_MANY_TO_MANY = "large_many_to_many"


class ModelLoader(ForgeModel):
    """MBart50 model loader implementation for summarization."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LARGE_MANY_TO_MANY: LLMModelConfig(
            pretrained_model_name="facebook/mbart-large-50-many-to-many-mmt",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE_MANY_TO_MANY

    sample_text = "Hello, my dog is cute."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None

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
            model="mbart50",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load the tokenizer for the model.
        Args:
            dtype_override: Optional dtype to override the default dtype.
        Returns:
            Tokenizer: The tokenizer for the model
        """

        from transformers import AutoTokenizer

        # Initialize tokenizer with dtype_override if provided
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the MBart50 model instance for this instance's variant.
        Args:
            dtype_override: Optional dtype to override the default dtype.
        Returns:
            model: The loaded model instance
        """

        from transformers import FlaxMBartForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        model = FlaxMBartForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MBart50 model with this instance's variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the conditional generation task
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return inputs
