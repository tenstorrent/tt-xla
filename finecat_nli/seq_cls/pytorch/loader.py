# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FineCat NLI model loader implementation for sequence classification (NLI).
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
    """Available FineCat NLI model variants for sequence classification."""

    FINECAT_NLI_L = "FineCat_NLI_L"


class ModelLoader(ForgeModel):
    """FineCat NLI model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.FINECAT_NLI_L: ModelConfig(
            pretrained_model_name="dleemiller/finecat-nli-l",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FINECAT_NLI_L

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FineCat NLI",
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

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        predicted_class_id = logits.argmax(-1).item()
        labels = ["entailment", "neutral", "contradiction"]
        print(f"Predicted: {labels[predicted_class_id]}")
