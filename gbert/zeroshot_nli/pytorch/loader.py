# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GBERT model loader implementation for zero-shot NLI sequence classification.
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
    """Available GBERT model variants for zero-shot NLI."""

    GBERT_LARGE_ZEROSHOT_NLI = "Large_Zeroshot_NLI"


class ModelLoader(ForgeModel):
    """GBERT model loader implementation for zero-shot NLI classification."""

    _VARIANTS = {
        ModelVariant.GBERT_LARGE_ZEROSHOT_NLI: ModelConfig(
            pretrained_model_name="svalabs/gbert-large-zeroshot-nli",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GBERT_LARGE_ZEROSHOT_NLI

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GBERT",
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

        premise = "Ich habe ein Problem mit meinem Iphone das so schnell wie möglich gelöst werden muss"
        hypothesis = "In diesem Satz geht es um das Thema Handy."

        inputs = self.tokenizer(
            premise,
            hypothesis,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        predicted_class_id = logits.argmax(-1).item()
        labels = ["contradiction", "neutral", "entailment"]
        print(f"Predicted: {labels[predicted_class_id]}")
