# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Electra model loader implementation for causal language modeling.
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
    """Available ELECTRA model variants."""

    BASE_DISCRIMINATOR = "base-discriminator"
    BASE_GENERATOR = "base-generator"
    LARGE_DISCRIMINATOR = "large-discriminator"
    SMALL_DISCRIMINATOR = "small-discriminator"


class ModelLoader(ForgeModel):
    """ELECTRA model loader implementation for causal language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_DISCRIMINATOR: LLMModelConfig(
            pretrained_model_name="google/electra-base-discriminator",
        ),
        ModelVariant.BASE_GENERATOR: LLMModelConfig(
            pretrained_model_name="google/electra-base-generator",
        ),
        ModelVariant.LARGE_DISCRIMINATOR: LLMModelConfig(
            pretrained_model_name="google/electra-large-discriminator",
        ),
        ModelVariant.SMALL_DISCRIMINATOR: LLMModelConfig(
            pretrained_model_name="google/electra-small-discriminator",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_DISCRIMINATOR

    # Shared configuration parameters
    sample_text = "Hello, my dog is cute"

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
            model="electra",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.

        Returns:
            tokenizer: The loaded tokenizer instance
        """

        from transformers import AutoTokenizer

        # Initialize tokenizer with dtype_override if provided
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        # Load the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the ELECTRA model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            model: The loaded model instance
        """

        from transformers import FlaxElectraForCausalLM

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        model = FlaxElectraForCausalLM.from_pretrained(self._model_name, **model_kwargs)

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ELECTRA model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return inputs
