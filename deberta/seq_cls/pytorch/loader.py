# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeBERTa model loader implementation for sequence classification.
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
    """Available DeBERTa model variants for sequence classification."""

    DEBERTA_XLARGE_MNLI = "XLarge_MNLI"
    DEBERTA_V3_XSMALL_MNLI_FEVER_ANLI_LING_BINARY = (
        "v3_xsmall_mnli_fever_anli_ling_binary"
    )


class ModelLoader(ForgeModel):
    """DeBERTa model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.DEBERTA_XLARGE_MNLI: ModelConfig(
            pretrained_model_name="microsoft/deberta-xlarge-mnli",
        ),
        ModelVariant.DEBERTA_V3_XSMALL_MNLI_FEVER_ANLI_LING_BINARY: ModelConfig(
            pretrained_model_name="MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEBERTA_XLARGE_MNLI

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DeBERTa",
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

    def _is_nli_variant(self):
        return self._variant == ModelVariant.DEBERTA_XLARGE_MNLI

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        if self._is_nli_variant():
            premise = "A man is eating food."
            hypothesis = "A man is eating a meal."
            inputs = self.tokenizer(
                premise,
                hypothesis,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            text = "This is a sample text for classification."
            inputs = self.tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        logits = co_out[0]
        predicted_class_id = logits.argmax(-1).item()
        if self._variant == ModelVariant.DEBERTA_V3_XSMALL_MNLI_FEVER_ANLI_LING_BINARY:
            labels = ["not_entailment", "entailment"]
        else:
            labels = ["contradiction", "neutral", "entailment"]
        print(f"Predicted: {labels[predicted_class_id]}")
