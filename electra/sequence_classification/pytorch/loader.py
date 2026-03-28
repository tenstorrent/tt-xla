# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ELECTRA model loader implementation for sequence classification task.
"""

from transformers import ElectraForSequenceClassification, ElectraTokenizerFast
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available ELECTRA sequence classification model variants."""

    CIRCULUS_KOELECTRA_EMOTION_V1 = "circulus_KoElectra_Emotion_v1"


class ModelLoader(ForgeModel):
    """ELECTRA model loader implementation for sequence classification task."""

    _VARIANTS = {
        ModelVariant.CIRCULUS_KOELECTRA_EMOTION_V1: LLMModelConfig(
            pretrained_model_name="circulus/koelectra-emotion-v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CIRCULUS_KOELECTRA_EMOTION_V1

    _SAMPLE_TEXTS = {
        ModelVariant.CIRCULUS_KOELECTRA_EMOTION_V1: "오늘 정말 기분이 좋아요",
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.review = self._SAMPLE_TEXTS.get(self._variant, "오늘 정말 기분이 좋아요")
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ELECTRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = ElectraTokenizerFast.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = ElectraForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model.eval()
        return self.model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.review,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        predicted_value = co_out[0].argmax(-1).item()
        print(f"Predicted Emotion: {self.model.config.id2label[predicted_value]}")
