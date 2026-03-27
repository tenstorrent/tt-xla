# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XLM-RoBERTa model loader implementation for sequence classification (sentiment analysis).
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """Available XLM-RoBERTa sequence classification model variants."""

    TWITTER_XLM_ROBERTA_BASE_SENTIMENT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    TEXTDETOX_XLMR_LARGE_TOXICITY_CLASSIFIER = (
        "textdetox/xlmr-large-toxicity-classifier"
    )


class ModelLoader(ForgeModel):
    """XLM-RoBERTa model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.TWITTER_XLM_ROBERTA_BASE_SENTIMENT: LLMModelConfig(
            pretrained_model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            max_length=128,
        ),
        ModelVariant.TEXTDETOX_XLMR_LARGE_TOXICITY_CLASSIFIER: LLMModelConfig(
            pretrained_model_name="textdetox/xlmr-large-toxicity-classifier",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TWITTER_XLM_ROBERTA_BASE_SENTIMENT

    _SAMPLE_TEXTS = {
        ModelVariant.TWITTER_XLM_ROBERTA_BASE_SENTIMENT: "Great road trip views! @ Shartlesville, Pennsylvania",
        ModelVariant.TEXTDETOX_XLMR_LARGE_TOXICITY_CLASSIFIER: "This is a friendly message.",
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.text = self._SAMPLE_TEXTS.get(
            self._variant, "Great road trip views! @ Shartlesville, Pennsylvania"
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="XLM-RoBERTa",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load XLM-RoBERTa model for sequence classification from Hugging Face."""

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
        """Prepare sample input for XLM-RoBERTa sequence classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        """Decode the model output for sequence classification."""
        predicted_class_id = co_out[0].argmax().item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_category = model.config.id2label[predicted_class_id]
            print(f"Predicted Sentiment: {predicted_category}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
