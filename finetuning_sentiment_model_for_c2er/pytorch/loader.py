# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Finetuning sentiment model for C2ER loader implementation for text classification.
"""

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from typing import Optional

from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available model variants for finetuning-sentiment-model-for-c2er."""

    FINETUNING_SENTIMENT_MODEL_FOR_C2ER = "finetuning-sentiment-model-for-c2er"


class ModelLoader(ForgeModel):
    """Finetuning sentiment model for C2ER loader for text classification."""

    _VARIANTS = {
        ModelVariant.FINETUNING_SENTIMENT_MODEL_FOR_C2ER: LLMModelConfig(
            pretrained_model_name="teomotun/finetuning-sentiment-model-for-c2er",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FINETUNING_SENTIMENT_MODEL_FOR_C2ER

    _SAMPLE_TEXTS = {
        ModelVariant.FINETUNING_SENTIMENT_MODEL_FOR_C2ER: "I love this product, it works great!",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.max_length = self._variant_config.max_length

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FinetuningsentimentmodelforC2ER",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = DistilBertForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        review = self._SAMPLE_TEXTS.get(
            self._variant, "I love this product, it works great!"
        )

        inputs = self.tokenizer(
            review,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        predicted_class_id = co_out[0].argmax().item()
        if (
            framework_model
            and hasattr(framework_model, "config")
            and hasattr(framework_model.config, "id2label")
        ):
            predicted_category = framework_model.config.id2label[predicted_class_id]
            print(f"Predicted category: {predicted_category}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
