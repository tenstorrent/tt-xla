# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KoELECTRA model loader implementation for sequence classification (wellness detection).
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available KoELECTRA model variants for sequence classification."""

    WELLNESS_V1 = "wellness-v1"


class ModelLoader(ForgeModel):
    """KoELECTRA model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.WELLNESS_V1: LLMModelConfig(
            pretrained_model_name="circulus/koelectra-wellness-v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WELLNESS_V1

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.model = None
        self.sample_text = "요즘 스트레스를 많이 받아서 힘들어요"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="KoELECTRA",
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

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model.eval()
        return self.model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        predicted_class_id = co_out[0].argmax().item()
        if (
            self.model
            and hasattr(self.model, "config")
            and hasattr(self.model.config, "id2label")
        ):
            predicted_label = self.model.config.id2label[predicted_class_id]
            print(f"Predicted label: {predicted_label}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
