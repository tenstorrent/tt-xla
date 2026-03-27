# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MarianMT model loader implementation for text translation.
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
    """Available MarianMT model variants for text translation."""

    OPUS_MT_TR_EN = "Opus_Mt_Tr_En"
    OPUS_MT_EN_AR = "Opus_Mt_En_Ar"
    OPUS_MT_DA_EN = "Opus_Mt_Da_En"


class ModelLoader(ForgeModel):
    """MarianMT model loader implementation for text translation."""

    _VARIANTS = {
        ModelVariant.OPUS_MT_TR_EN: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-tr-en",
        ),
        ModelVariant.OPUS_MT_EN_AR: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-en-ar",
        ),
        ModelVariant.OPUS_MT_DA_EN: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-da-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPUS_MT_TR_EN

    _SAMPLE_TEXTS = {
        ModelVariant.OPUS_MT_TR_EN: "Merhaba dünya, bugün hava çok güzel.",
        ModelVariant.OPUS_MT_EN_AR: "My friends are cool but they eat too many carbs.",
        ModelVariant.OPUS_MT_DA_EN: "Hej verden, i dag er vejret meget smukt.",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._tokenizer = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="MarianMT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        from transformers import MarianTokenizer

        self._tokenizer = MarianTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MarianMT model instance for this instance's variant."""
        from transformers import MarianMTModel

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MarianMTModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MarianMT model."""
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        sample_text = self._SAMPLE_TEXTS.get(self._variant)
        inputs = self._tokenizer(
            sample_text,
            return_tensors="pt",
        )

        decoder_start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
