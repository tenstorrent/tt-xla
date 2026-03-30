# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ModernBERT model loader implementation for sequence classification.
"""
from typing import Optional

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
    """Available ModernBERT model variants for sequence classification."""

    GO_EMOTIONS_BASE = "Go_Emotions_Base"
    SIMILARITY_CLASSIFIER_F168 = "Similarity_Classifier_F168"


class ModelLoader(ForgeModel):
    """ModernBERT model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.GO_EMOTIONS_BASE: LLMModelConfig(
            pretrained_model_name="cirimus/modernbert-base-go-emotions",
            max_length=128,
        ),
        ModelVariant.SIMILARITY_CLASSIFIER_F168: LLMModelConfig(
            pretrained_model_name="dogtooth/similarity-classifier-f168-hf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GO_EMOTIONS_BASE

    _SAMPLE_TEXTS = {
        ModelVariant.GO_EMOTIONS_BASE: (
            "I am so happy and excited about this opportunity!",
        ),
        ModelVariant.SIMILARITY_CLASSIFIER_F168: (
            "The cat sat on the mat.",
            "A feline was resting on the rug.",
        ),
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ModernBERT",
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
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        sample = self._SAMPLE_TEXTS.get(
            self._variant, ("I am so happy and excited about this opportunity!",)
        )
        text_args = sample

        inputs = self.tokenizer(
            *text_args,
            max_length=self._variant_config.max_length,
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
            predicted_label = framework_model.config.id2label[predicted_class_id]
            print(f"Predicted label: {predicted_label}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
