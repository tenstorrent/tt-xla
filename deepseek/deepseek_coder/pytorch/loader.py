# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Coder model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from unittest.mock import patch
import os
from ....tools.utils import generate_no_cache, pad_inputs
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
from transformers.dynamic_module_utils import get_imports


class ModelVariant(StrEnum):
    """Available DeepSeek Coder model variants."""

    DEEPSEEK_1_3B_INSTRUCT = "1_3b_instruct"


class ModelLoader(ForgeModel):
    """DeepSeek Coder model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEEPSEEK_1_3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="deepseek-ai/deepseek-coder-1.3b-instruct",
            max_length=2048,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_1_3B_INSTRUCT

    # Sample prompt text
    sample_text = "write a bubble sort algorithm in python."

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

        Returns:
            ModelInfo: Information about the model and variant.
        """
        return ModelInfo(
            model="deepseek_coder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the DeepSeek Coder model instance."""

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the DeepSeek Coder model."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        padded_inputs, seq_len = pad_inputs(inputs)

        return padded_inputs, seq_len

    def decode_output(self, max_new_tokens, model, inputs, seq_len, tokenizer):
        """Generates text .

        Args:
            max_new_tokens (int): The maximum number of new tokens to generate.
            model (torch.nn.Module): The language model used for token generation.
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len), representing tokenized text.
            seq_len (int): The current sequence length before generation starts.
            tokenizer: The tokenizer used to decode token IDs into text.

        """
        generated_text = generate_no_cache(
            max_new_tokens, model, inputs, seq_len, tokenizer
        )
        return generated_text
