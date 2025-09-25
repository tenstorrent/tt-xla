# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T5 model loader implementation
"""

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available T5 model variants."""

    SMALL = "t5-small"
    BASE = "t5-base"
    LARGE = "t5-large"
    FLAN_T5_SMALL = "google/flan-t5-small"
    FLAN_T5_BASE = "google/flan-t5-base"
    FLAN_T5_LARGE = "google/flan-t5-large"


class ModelLoader(ForgeModel):
    """T5 model loader implementation for conditional generation tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.SMALL: LLMModelConfig(
            pretrained_model_name="t5-small",
            max_length=512,
        ),
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="t5-base",
            max_length=512,
        ),
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="t5-large",
            max_length=512,
        ),
        ModelVariant.FLAN_T5_SMALL: LLMModelConfig(
            pretrained_model_name="google/flan-t5-small",
            max_length=512,
        ),
        ModelVariant.FLAN_T5_BASE: LLMModelConfig(
            pretrained_model_name="google/flan-t5-base",
            max_length=512,
        ),
        ModelVariant.FLAN_T5_LARGE: LLMModelConfig(
            pretrained_model_name="google/flan-t5-large",
            max_length=512,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SMALL

    # Shared configuration parameters
    sample_text = """summarize: Researchers have extensively studied the benefits of having pets,
                    particularly dogs, on human health and well-being. Findings suggest that pet ownership
                    can lead to improved mental health, reduced stress levels, and even physical health benefits
                    such as lower blood pressure and increased physical activity levels due to regular walks."""

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

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
            model="t5",
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the T5 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The T5 model instance for conditional generation.
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

        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, use_cache=False, **model_kwargs
        )
        model.eval()
        # Cache model for use in load_inputs (to avoid reloading)
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the T5 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # T5 requires decoder input ids also as an input
        decoder_start_token_tensor = torch.tensor(
            self._cached_model.generation_config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
