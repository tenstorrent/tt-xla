# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Robust Swedish Sentiment Multiclass model loader implementation for sequence classification.

Loads the KBLab/robust-swedish-sentiment-multiclass model for Swedish
multi-class sentiment analysis (NEGATIVE, NEUTRAL, POSITIVE).
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

REPO_ID = "KBLab/robust-swedish-sentiment-multiclass"


class ModelVariant(StrEnum):
    """Available Robust Swedish Sentiment model variants."""

    KBLAB_ROBUST_SWEDISH_SENTIMENT_MULTICLASS = (
        "kblab_robust_swedish_sentiment_multiclass"
    )


class ModelLoader(ForgeModel):
    """Robust Swedish Sentiment Multiclass model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.KBLAB_ROBUST_SWEDISH_SENTIMENT_MULTICLASS: LLMModelConfig(
            pretrained_model_name=REPO_ID,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KBLAB_ROBUST_SWEDISH_SENTIMENT_MULTICLASS

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.text = "Det här var en fantastisk film!"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="Robust_Swedish_Sentiment",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Robust Swedish Sentiment model for sequence classification."""

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
        """Prepare sample input for Swedish sentiment classification."""
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
