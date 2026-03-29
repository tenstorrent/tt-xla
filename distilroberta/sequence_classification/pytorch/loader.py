# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilRoBERTa model loader implementation for sequence classification.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available DistilRoBERTa model variants for sequence classification."""

    EMOTION_ENGLISH_BASE = "Emotion_English_Base"
    PROTECTAI_REJECTION_V1 = "ProtectAI_Rejection_V1"


class ModelLoader(ForgeModel):
    """DistilRoBERTa model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.EMOTION_ENGLISH_BASE: LLMModelConfig(
            pretrained_model_name="j-hartmann/emotion-english-distilroberta-base",
            max_length=128,
        ),
        ModelVariant.PROTECTAI_REJECTION_V1: LLMModelConfig(
            pretrained_model_name="protectai/distilroberta-base-rejection-v1",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EMOTION_ENGLISH_BASE

    _SAMPLE_TEXTS = {
        ModelVariant.EMOTION_ENGLISH_BASE: "I am so happy today, everything is going great!",
        ModelVariant.PROTECTAI_REJECTION_V1: "Sorry, but I can't assist with that request.",
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = self._SAMPLE_TEXTS.get(
            self._variant, "I am so happy today, everything is going great!"
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="DistilRoBERTa",
            variant=variant_name,
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
