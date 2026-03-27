# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-2 model loader implementation for causal language modeling.
"""
from transformers import GPT2LMHeadModel, AutoTokenizer
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
    """Available GPT-2 causal LM model variants."""

    TINY_RANDOM = "tiny-random"


class ModelLoader(ForgeModel):
    """GPT-2 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.TINY_RANDOM: LLMModelConfig(
            pretrained_model_name="peft-internal-testing/tiny-random-GPT2LMHeadModel",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    # Shared configuration parameters
    sample_text = "Hello there fellow traveler"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

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
            model="GPT-2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GPT-2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The GPT-2 model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name, **model_kwargs)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the GPT-2 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the model [input_ids, attention_mask].
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the causal language modeling task
        input_tokens = self.tokenizer(
            self.sample_text,
            max_length=self._variant_config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Return as list of tensors for the wrapper
        return [input_tokens["input_ids"], input_tokens["attention_mask"]]
