# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WangchanBERTa model loader implementation for sequence classification (sentiment analysis).
WangchanBERTa is a Thai-language CamemBERT/RoBERTa model fine-tuned for sentiment analysis.
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
    """Available WangchanBERTa model variants for sequence classification."""

    FINETUNED_SENTIMENT = "Finetuned_Sentiment"


class ModelLoader(ForgeModel):
    """WangchanBERTa model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.FINETUNED_SENTIMENT: LLMModelConfig(
            pretrained_model_name="poom-sci/WangchanBERTa-finetuned-sentiment",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FINETUNED_SENTIMENT

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = "วันนี้อากาศดีมาก ฉันมีความสุข"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="WangchanBERTa",
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

    def decode_output(self, co_out, framework_model=None):
        predicted_class_id = co_out[0].argmax().item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_sentiment = model.config.id2label[predicted_class_id]
            print(f"Predicted Sentiment: {predicted_sentiment}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
