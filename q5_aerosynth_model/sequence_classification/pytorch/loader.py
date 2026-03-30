# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Q5-AeroSynth model loader implementation for sequence classification.
"""
from typing import Optional

from transformers import BertForSequenceClassification, BertTokenizer

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Q5-AeroSynth model variants for sequence classification."""

    Q5_AEROSYNTH_MODEL = "Q5_AeroSynth_Model"


class ModelLoader(ForgeModel):
    """Q5-AeroSynth model loader implementation for sequence classification tasks."""

    _VARIANTS = {
        ModelVariant.Q5_AEROSYNTH_MODEL: LLMModelConfig(
            pretrained_model_name="TextAsData/Q5-AeroSynth-model",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q5_AEROSYNTH_MODEL

    _SAMPLE_TEXTS = {
        ModelVariant.Q5_AEROSYNTH_MODEL: "Please review the attached quarterly sales report and provide feedback.",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.review = self._SAMPLE_TEXTS.get(
            self._variant,
            "Please review the attached quarterly sales report and provide feedback.",
        )
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Q5-AeroSynth",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Q5-AeroSynth model for sequence classification from Hugging Face."""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BertForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Q5-AeroSynth sequence classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.review,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for sequence classification."""
        predicted_value = co_out[0].argmax(-1).item()
        print(f"Predicted Class: {self.model.config.id2label[predicted_value]}")
