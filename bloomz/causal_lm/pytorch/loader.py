# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
BLOOMZ model loader implementation for causal language modeling.
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


class ModelVariant(StrEnum):
    """Available BLOOMZ model variants."""

    BLOOMZ_560M = "560M"
    BLOOMZ_1B1 = "1b1"
    BLOOMZ_1B7 = "1b7"
    BLOOMZ_3B = "3B"
    BLOOMZ_7B = "7B"


class ModelLoader(ForgeModel):
    """BLOOMZ model loader implementation for causal language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BLOOMZ_560M: LLMModelConfig(
            pretrained_model_name="bigscience/bloomz-560m",
        ),
        ModelVariant.BLOOMZ_1B1: LLMModelConfig(
            pretrained_model_name="bigscience/bloomz-1b1",
        ),
        ModelVariant.BLOOMZ_1B7: LLMModelConfig(
            pretrained_model_name="bigscience/bloomz-1b7",
        ),
        ModelVariant.BLOOMZ_3B: LLMModelConfig(
            pretrained_model_name="bigscience/bloomz-3b",
        ),
        ModelVariant.BLOOMZ_7B: LLMModelConfig(
            pretrained_model_name="bigscience/bloomz-7b1",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BLOOMZ_3B

    sample_text = "Translate to English: Je t'aime."

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
        # Use the provided variant or fall back to default
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BLOOMZ",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
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
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BLOOMZ model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            model: The loaded model instance
        """
        from transformers import AutoModelForCausalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BLOOMZ model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the causal language modeling task
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        return inputs
