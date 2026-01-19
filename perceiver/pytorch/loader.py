# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Perceiver model loader implementation for masked language modeling
"""
import torch
from transformers import PerceiverForMaskedLM, PerceiverTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Perceiver model variants."""

    LANGUAGE_PERCEIVER = "deepmind/language-perceiver"


class ModelLoader(ForgeModel):
    """Perceiver model loader implementation for masked language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LANGUAGE_PERCEIVER: ModelConfig(
            pretrained_model_name="deepmind/language-perceiver",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LANGUAGE_PERCEIVER

    # Shared configuration parameters
    sample_text = "This is an incomplete sentence where some words are missing."

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
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="perceiver",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_MASKED_LM,
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
        # Initialize tokenizer
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = PerceiverTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Perceiver model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Perceiver model instance for masked language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = PerceiverForMaskedLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Perceiver model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Raw encoding dictionary with input_ids and attention_mask.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs
        encoding = self.tokenizer(
            self.sample_text, padding="max_length", return_tensors="pt"
        )

        return encoding

    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded answer text for the masked tokens
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs[0]
        masked_tokens_predictions = logits[0, 51:61].argmax(dim=-1)

        return self.tokenizer.decode(masked_tokens_predictions)
