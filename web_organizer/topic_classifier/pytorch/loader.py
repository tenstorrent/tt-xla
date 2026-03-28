# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WebOrganizer TopicClassifier model loader implementation for sequence classification.
"""
from typing import Optional

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available WebOrganizer TopicClassifier model variants."""

    TOPIC_CLASSIFIER = "TopicClassifier"


class ModelLoader(ForgeModel):
    """WebOrganizer TopicClassifier model loader implementation."""

    _VARIANTS = {
        ModelVariant.TOPIC_CLASSIFIER: ModelConfig(
            pretrained_model_name="WebOrganizer/TopicClassifier",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TOPIC_CLASSIFIER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WebOrganizer_TopicClassifier",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            use_memory_efficient_attention=False,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        sample_text = (
            "http://www.example.com\n\n"
            "How to build a computer from scratch? "
            "Here are the components you need to get started."
        )

        inputs = self.tokenizer(
            sample_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        predicted_class_id = logits.argmax(-1).item()
        labels = [
            "Adult",
            "Art & Design",
            "Software Dev.",
            "Crime & Law",
            "Education & Jobs",
            "Hardware",
            "Entertainment",
            "Social Life",
            "Fashion & Beauty",
            "Finance & Business",
            "Food & Dining",
            "Games",
            "Health",
            "History",
            "Home & Hobbies",
            "Industrial",
            "Literature",
            "Politics",
            "Religion",
            "Science & Tech.",
            "Software",
            "Sports & Fitness",
            "Transportation",
            "Travel",
        ]
        print(f"Predicted Topic: {labels[predicted_class_id]}")
