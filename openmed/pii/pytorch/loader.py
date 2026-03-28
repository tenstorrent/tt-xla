# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed PII ClinicalBGE model loader implementation for PII detection in clinical text.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
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
    """Available OpenMed PII ClinicalBGE model variants."""

    CLINICAL_BGE_LARGE_335M_V1 = "ClinicalBGE-Large-335M-v1"


class ModelLoader(ForgeModel):
    """OpenMed PII ClinicalBGE model loader for PII detection in clinical text."""

    _VARIANTS = {
        ModelVariant.CLINICAL_BGE_LARGE_335M_V1: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CLINICAL_BGE_LARGE_335M_V1

    _SAMPLE_TEXTS = {
        ModelVariant.CLINICAL_BGE_LARGE_335M_V1: "Patient John Smith, SSN 123-45-6789, was admitted to the clinic on 2024-01-15.",
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = self._SAMPLE_TEXTS.get(
            self._variant,
            "Patient John Smith, SSN 123-45-6789, was admitted to the clinic on 2024-01-15.",
        )
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "pii"
        return ModelInfo(
            model="OpenMed",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
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
        inputs = self.load_inputs()
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )
        predicted_tokens_classes = [
            self.model.config.id2label[t.item()] for t in predicted_token_class_ids
        ]

        print(f"Context: {self.sample_text}")
        print(f"PII Tags: {predicted_tokens_classes}")
