# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed PII French model loader implementation for token classification.
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
    """Available OpenMed PII French model variants for token classification."""

    OPENMED_PII_FRENCH_BIOCLINICALBERT_BASE_110M_V1 = (
        "OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1"
    )


class ModelLoader(ForgeModel):
    """OpenMed PII French model loader for token classification."""

    _VARIANTS = {
        ModelVariant.OPENMED_PII_FRENCH_BIOCLINICALBERT_BASE_110M_V1: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_PII_FRENCH_BIOCLINICALBERT_BASE_110M_V1

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "Dr. Jean Dupont (NSS: 1 85 12 75 108 123 45) peut etre contacte a jean.dupont@hopital.fr ou au 06 12 34 56 78."
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "PII_French_BioClinicalBERT_Base_110M_v1"
        return ModelInfo(
            model="OpenMed PII French",
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
