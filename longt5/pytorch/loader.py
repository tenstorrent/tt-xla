# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongT5 model loader implementation for Conditional Generation.
"""

import torch
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
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
    """Available LongT5 model variants."""

    BASE_TGLOBAL = "Base_Tglobal"
    LARGE_LOCAL = "Large_Local"
    XL_TGLOBAL = "Xl_Tglobal"


class ModelLoader(ForgeModel):
    """LongT5 model loader implementation for conditional generation tasks."""

    _VARIANTS = {
        ModelVariant.BASE_TGLOBAL: LLMModelConfig(
            pretrained_model_name="google/long-t5-tglobal-base",
        ),
        ModelVariant.LARGE_LOCAL: LLMModelConfig(
            pretrained_model_name="google/long-t5-local-large",
        ),
        ModelVariant.XL_TGLOBAL: LLMModelConfig(
            pretrained_model_name="google/long-t5-tglobal-xl",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_TGLOBAL

    sample_text = "summarize: My friends are cool but they eat too many carbs."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

        super().__init__(variant)
        self._tokenizer = None
        self._cached_model = None
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
            model="LongT5",
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

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LongT5 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The LongT5 model instance for conditional generation.
        """

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LongT5ForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the LongT5 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_tensor = torch.tensor(
            self._cached_model.generation_config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
