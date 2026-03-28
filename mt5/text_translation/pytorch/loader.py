# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MT5 model loader implementation for text translation.
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
    """Available MT5 model variants for text translation."""

    PARSINLU_OPUS_TRANSLATION_FA_EN = "Parsinlu_Opus_Translation_Fa_En"


class ModelLoader(ForgeModel):
    """MT5 model loader implementation for text translation."""

    _VARIANTS = {
        ModelVariant.PARSINLU_OPUS_TRANSLATION_FA_EN: LLMModelConfig(
            pretrained_model_name="persiannlp/mt5-small-parsinlu-opus-translation_fa_en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PARSINLU_OPUS_TRANSLATION_FA_EN

    sample_text = "ستایش خدای را که پروردگار جهانیان است."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._tokenizer = None
        self._model = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="mT5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        from transformers import MT5Tokenizer

        self._tokenizer = MT5Tokenizer.from_pretrained(self._model_name)

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MT5 model instance for this instance's variant."""
        from transformers import MT5ForConditionalGeneration

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MT5ForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )
        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MT5 model."""
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
