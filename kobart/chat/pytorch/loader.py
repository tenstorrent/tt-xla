# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
KoBART model loader implementation for chat generation.
"""

import torch
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
    """Available KoBART chat model variants."""

    ALL_V2 = "All_V2"


class ModelLoader(ForgeModel):
    """KoBART model loader implementation for Korean chat generation."""

    _VARIANTS = {
        ModelVariant.ALL_V2: LLMModelConfig(
            pretrained_model_name="circulus/kobart-chat-all-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALL_V2

    sample_text = "안녕하세요, 오늘 날씨가 어때요?"

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
            model="KoBART",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.

        Returns:
            tokenizer: The loaded tokenizer instance
        """
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the KoBART model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        from transformers import BartForConditionalGeneration

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BartForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the KoBART model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        from transformers.models.bart.modeling_bart import shift_tokens_right

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        model = self.load_model(dtype_override=dtype_override)
        decoder_input_ids = shift_tokens_right(
            inputs["input_ids"],
            model.config.pad_token_id,
            model.config.decoder_start_token_id,
        )

        result = [
            inputs["input_ids"],
            inputs["attention_mask"],
            decoder_input_ids,
        ]

        if dtype_override is not None:
            result = [
                t.to(dtype_override) if t.dtype == torch.float32 else t for t in result
            ]

        return result
