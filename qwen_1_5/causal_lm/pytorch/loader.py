# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 1.5 model loader implementation for causal language modeling.
"""
import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
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
    """Available Qwen 1.5 model variants for causal language modeling."""

    QWEN_1_5_0_5B = "0_5b"
    QWEN_1_5_0_5B_CHAT = "0_5b_chat"


class ModelLoader(ForgeModel):
    """Qwen 1.5 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_1_5_0_5B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen1.5-0.5B",
            max_length=128,
        ),
        ModelVariant.QWEN_1_5_0_5B_CHAT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen1.5-0.5B-Chat",
            max_length=512,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_1_5_0_5B

    # Shared configuration parameters
    sample_text = "My name is Jim Keller and"
    chat_messages = [
        {"role": "system", "content": "You are Jim Keller, the CEO of Tenstorrent"},
        {"role": "user", "content": "Introduce yourself please!"},
    ]

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
            model="qwen_1_5",
            variant=variant,
            group=ModelGroup.GENERALITY,
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
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        # Set pad token to eos token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Qwen 1.5 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen 1.5 model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = Qwen2ForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)

        # Disable DynamicCache
        model._supports_cache_class = False
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 1.5 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        # Auto-detect if we should use chat template based on variant name
        use_chat_template = "chat" in str(self._variant).lower()

        from loguru import logger

        logger.info("use_chat_template ={}", use_chat_template)

        if use_chat_template:
            # Use chat template
            batch_messages = [self.chat_messages] * batch_size
            prompts = [
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in batch_messages
            ]
        else:
            # Use regular text
            prompts = [self.sample_text] * batch_size

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return inputs
