# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MMS TTS model loader implementation for text-to-speech tasks using VITS architecture.
"""

from typing import Optional

from transformers import AutoTokenizer, VitsModel

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MMS TTS model variants."""

    BENGALI = "Bengali"
    KINYARWANDA = "Kinyarwanda"
    TELUGU = "Telugu"


class ModelLoader(ForgeModel):
    """MMS TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.BENGALI: ModelConfig(
            pretrained_model_name="facebook/mms-tts-ben",
        ),
        ModelVariant.KINYARWANDA: ModelConfig(
            pretrained_model_name="facebook/mms-tts-kin",
        ),
        ModelVariant.TELUGU: ModelConfig(
            pretrained_model_name="facebook/mms-tts-tel",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KINYARWANDA

    _SAMPLE_TEXTS = {
        ModelVariant.BENGALI: "আমাদের সিস্টেম ব্যবহার করার জন্য স্বাগতম।",
        ModelVariant.KINYARWANDA: "Muraho, murakaza neza mu gukoresha sisitemu yacu.",
        ModelVariant.TELUGU: "మా వ్యవస్థను ఉపయోగించినందుకు స్వాగతం.",
    }

    sample_text = _SAMPLE_TEXTS[DEFAULT_VARIANT]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MMS_TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VitsModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        sample_text = self._SAMPLE_TEXTS.get(self._variant, self.sample_text)
        inputs = self._tokenizer(sample_text, return_tensors="pt")

        return inputs
