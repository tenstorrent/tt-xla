# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpanMarker NER model loader implementation for named entity recognition.
"""

import torch
from span_marker import SpanMarkerModel
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
    """Available SpanMarker NER model variants."""

    GENERIC_NER_V1_FEWNERD_FINE_SUPER = "generic-ner-v1-fewnerd-fine-super"


class ModelLoader(ForgeModel):
    """SpanMarker NER model loader for named entity recognition."""

    _VARIANTS = {
        ModelVariant.GENERIC_NER_V1_FEWNERD_FINE_SUPER: LLMModelConfig(
            pretrained_model_name="guishe/span-marker-generic-ner-v1-fewnerd-fine-super",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERIC_NER_V1_FEWNERD_FINE_SUPER

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = (
            'Most of the Steven Seagal movie "Under Siege" was filmed on Mobile Bay.'
        )
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SpanMarker NER",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model = SpanMarkerModel.from_pretrained(self.model_name)
        self.tokenizer = model.tokenizer
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        self.model = model
        return model

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
        entities = self.model.predict(self.sample_text)
        print(f"Context: {self.sample_text}")
        print(f"Entities: {entities}")
