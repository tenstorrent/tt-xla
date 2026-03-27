# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeBERTa V3 Base model loader implementation for zero-shot classification.
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
    """Available DeBERTa model variants for zero-shot classification."""

    DEBERTA_V3_BASE_ZEROSHOT_V2 = "V3_Base_Zeroshot_v2.0"


class ModelLoader(ForgeModel):
    """DeBERTa V3 Base model loader for zero-shot classification."""

    _VARIANTS = {
        ModelVariant.DEBERTA_V3_BASE_ZEROSHOT_V2: ModelConfig(
            pretrained_model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEBERTA_V3_BASE_ZEROSHOT_V2

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

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        text = "Angela Merkel is a politician in Germany and target of a conspiracy theory."
        hypothesis = "This text is about politics."

        inputs = self.tokenizer(
            text,
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
        labels = ["entailment", "not_entailment"]
        print(f"Predicted: {labels[predicted_class_id]}")
