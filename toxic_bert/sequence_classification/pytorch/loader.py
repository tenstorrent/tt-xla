# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Toxic BERT model loader implementation for sequence classification.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Toxic BERT model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Toxic BERT model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="unitary/toxic-bert",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = 128
        self.tokenizer = None
        self.sample_text = (
            "I really enjoyed meeting new people from different cultures."
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="Toxic_BERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
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

    def load_inputs(self):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        predicted_class_id = co_out[0].argmax(-1).item()
        label = self.model.config.id2label[predicted_class_id]
        print(f"Predicted Label: {label}")
