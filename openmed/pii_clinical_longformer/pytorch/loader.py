# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed PII ClinicalLongformer model loader implementation for token classification.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
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
    """Available OpenMed PII ClinicalLongformer model variants."""

    OPENMED_PII_CLINICALLONGFORMER_BASE_149M_V1 = (
        "OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1"
    )


class ModelLoader(ForgeModel):
    """OpenMed PII ClinicalLongformer model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.OPENMED_PII_CLINICALLONGFORMER_BASE_149M_V1: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_PII_CLINICALLONGFORMER_BASE_149M_V1

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "Dr. John Smith treated patient Jane Doe at City Hospital on 03/15/2024. Contact: john.smith@hospital.org, SSN 123-45-6789."
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="OpenMed PII ClinicalLongformer",
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
        print(f"Answer: {predicted_tokens_classes}")
