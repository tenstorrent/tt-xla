# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Translation PT-EN T5 model loader implementation for Portuguese to English translation.
"""

import torch
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
    """Available Translation PT-EN T5 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Translation PT-EN T5 model loader implementation for Portuguese to English translation."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="unicamp-dl/translation-pt-en-t5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_text = "translate Portuguese to English: Eu gosto de comer arroz."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._tokenizer = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="Translation_PT_EN_T5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Translation PT-EN T5 model instance."""
        from transformers import AutoModelForSeq2SeqLM

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Translation PT-EN T5 model."""
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
