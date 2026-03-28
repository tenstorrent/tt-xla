# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilCamemBERT NLI model loader for sequence classification.
DistilCamemBERT is a distilled French RoBERTa model fine-tuned on XNLI for
natural language inference (entailment, neutral, contradiction).
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
    """Available DistilCamemBERT NLI model variants."""

    DISTILCAMEMBERT_BASE_NLI = "Base_NLI"


class ModelLoader(ForgeModel):
    """DistilCamemBERT NLI model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.DISTILCAMEMBERT_BASE_NLI: ModelConfig(
            pretrained_model_name="cmarkea/distilcamembert-base-nli",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILCAMEMBERT_BASE_NLI

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DistilCamemBERT NLI",
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

        premise = "Le style tres cinephile de Quentin Tarantino est un homme qui a fait un film."
        hypothesis = "Ce texte parle de cinema."

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
        labels = ["contradiction", "entailment", "neutral"]
        print(f"Predicted: {labels[predicted_class_id]}")
