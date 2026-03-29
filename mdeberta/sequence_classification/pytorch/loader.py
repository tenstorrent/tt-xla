# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mDeBERTa V3 model loader implementation for sequence classification.

Uses GroNLP/mdebertav3-subjectivity-multilingual, a multilingual subjectivity
detection model fine-tuned from microsoft/mdeberta-v3-base. It classifies
sentences as subjective (SUBJ) or objective (OBJ).
"""
from typing import Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    """Available mDeBERTa V3 model variants for sequence classification."""

    GRONLP_MDEBERTAV3_SUBJECTIVITY_MULTILINGUAL = (
        "GroNLP_mDeBERTaV3_Subjectivity_Multilingual"
    )


class ModelLoader(ForgeModel):
    """mDeBERTa V3 model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.GRONLP_MDEBERTAV3_SUBJECTIVITY_MULTILINGUAL: ModelConfig(
            pretrained_model_name="GroNLP/mdebertav3-subjectivity-multilingual",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRONLP_MDEBERTAV3_SUBJECTIVITY_MULTILINGUAL

    sample_text = "The president announced new policies today."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mDeBERTa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
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
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        inputs = self.tokenizer(
            self.sample_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        predicted_class_id = logits.argmax(-1).item()
        predicted_label = self.model.config.id2label[predicted_class_id]
        print(f"Predicted: {predicted_label}")
