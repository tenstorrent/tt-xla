# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ELECTRA model loader implementation for sequence classification (sentiment analysis).
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from third_party.tt_forge_models.config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available ELECTRA sequence classification model variants."""

    CIRCULUS_KOELECTRA_SENTIMENT_V1 = "circulus/koelectra-sentiment-v1"


class ModelLoader(ForgeModel):
    """ELECTRA model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.CIRCULUS_KOELECTRA_SENTIMENT_V1: LLMModelConfig(
            pretrained_model_name="circulus/koelectra-sentiment-v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CIRCULUS_KOELECTRA_SENTIMENT_V1

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "이 영화 정말 재미있고 감동적이었어요."
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        predicted_value = co_out[0].argmax(-1).item()
        print(f"Predicted Sentiment: {self.model.config.id2label[predicted_value]}")
