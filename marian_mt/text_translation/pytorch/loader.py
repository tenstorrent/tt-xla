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

    OPUS_MT_EN_DRA = "Opus_Mt_En_Dra"
    OPUS_MT_EN_HE = "Opus_Mt_En_He"
    OPUS_MT_EN_ID = "Opus_Mt_En_Id"
    OPUS_MT_FR_ES = "Opus_Mt_Fr_Es"
    OPUS_MT_NL_ES = "Opus_Mt_Nl_Es"
    OPUS_MT_ROA_EN = "Opus_Mt_Roa_En"
    OPUS_MT_TR_EN = "Opus_Mt_Tr_En"
    OPUS_MT_EN_EL = "Opus_Mt_En_El"
    OPUS_MT_EN_SV = "Opus_Mt_En_Sv"
    OPUS_MT_TH_EN = "Opus_Mt_Th_En"
    TINY_RANDOM_MARIAN = "Tiny_Random_Marian"
    TINY_MARIAN_EN_DE = "Tiny_Marian_En_De"


class ModelLoader(ForgeModel):
    """MarianMT model loader implementation for text translation."""

    _VARIANTS = {
        ModelVariant.OPUS_MT_EN_DRA: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-en-dra",
        ),
        ModelVariant.OPUS_MT_EN_HE: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-en-he",
        ),
        ModelVariant.OPUS_MT_EN_ID: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-en-id",
        ),
        ModelVariant.OPUS_MT_FR_ES: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-fr-es",
        ),
        ModelVariant.OPUS_MT_NL_ES: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-nl-es",
        ),
        ModelVariant.OPUS_MT_ROA_EN: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-roa-en",
        ),
        ModelVariant.OPUS_MT_TR_EN: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-tr-en",
        ),
        ModelVariant.OPUS_MT_EN_EL: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-en-el",
        ),
        ModelVariant.OPUS_MT_EN_SV: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-en-sv",
        ),
        ModelVariant.OPUS_MT_TH_EN: LLMModelConfig(
            pretrained_model_name="Helsinki-NLP/opus-mt-th-en",
        ),
        ModelVariant.TINY_RANDOM_MARIAN: LLMModelConfig(
            pretrained_model_name="optimum-internal-testing/tiny-random-marian",
        ),
        ModelVariant.TINY_MARIAN_EN_DE: LLMModelConfig(
            pretrained_model_name="sshleifer/tiny-marian-en-de",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPUS_MT_TR_EN

    _SAMPLE_TEXTS = {
        ModelVariant.OPUS_MT_TR_EN: "Merhaba dünya, bugün hava çok güzel.",
        ModelVariant.OPUS_MT_EN_EL: "My name is Wolfgang and I live in Berlin.",
        ModelVariant.OPUS_MT_EN_SV: "My name is Wolfgang and I live in Berlin.",
        ModelVariant.OPUS_MT_TH_EN: "สวัสดีครับ วันนี้อากาศดีมาก",
        ModelVariant.TINY_RANDOM_MARIAN: "My name is Wolfgang and I live in Berlin.",
        ModelVariant.TINY_MARIAN_EN_DE: "My name is Wolfgang and I live in Berlin.",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._tokenizer = None
        self._model = None
        self.sample_text = self._SAMPLE_TEXTS[self._variant]

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
        if self._model is None:
            self.load_model(dtype_override=dtype_override)

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
