# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpanMarker model loader implementation for named entity recognition.
"""

import torch
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    ModelConfig,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available SpanMarker model variants for NER."""

    ROBERTA_LARGE_ONTONOTES5 = "tomaarsen/span-marker-roberta-large-ontonotes5"


class ModelLoader(ForgeModel):
    """SpanMarker model loader implementation for named entity recognition."""

    _VARIANTS = {
        ModelVariant.ROBERTA_LARGE_ONTONOTES5: ModelConfig(
            pretrained_model_name="tomaarsen/span-marker-roberta-large-ontonotes5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBERTA_LARGE_ONTONOTES5

    def __init__(self, variant=None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.sample_text = "Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris."

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SpanMarker",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load SpanMarker model for NER from Hugging Face.

        Returns:
            torch.nn.Module: The SpanMarker model instance.
        """
        from span_marker import SpanMarkerModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = SpanMarkerModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for SpanMarker NER.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        tokenized = self.model.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return tokenized

    def decode_output(self, co_out):
        """Decode the model output for NER."""
        entities = self.model.predict(self.sample_text)

        print(f"Context: {self.sample_text}")
        print(f"Entities: {entities}")
