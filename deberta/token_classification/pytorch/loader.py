# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeBERTa model loader implementation for token classification (NER).
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from third_party.tt_forge_models.config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available DeBERTa token classification model variants."""

    OPENMED_NER_CHEMICALDETECT_SUPERCLINICAL_434M = (
        "OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-434M"
    )
    OPENMED_PII_GERMAN_SUPERCLINICAL_BASE_184M_V1 = (
        "OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1"
    )
    BLAZE999_MEDICAL_NER = "blaze999/Medical-NER"


class ModelLoader(ForgeModel):
    """DeBERTa model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.OPENMED_NER_CHEMICALDETECT_SUPERCLINICAL_434M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-434M",
            max_length=128,
        ),
        ModelVariant.OPENMED_PII_GERMAN_SUPERCLINICAL_BASE_184M_V1: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1",
            max_length=128,
        ),
        ModelVariant.BLAZE999_MEDICAL_NER: LLMModelConfig(
            pretrained_model_name="blaze999/Medical-NER",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_NER_CHEMICALDETECT_SUPERCLINICAL_434M

    def __init__(self, variant=None):
        super().__init__(variant)
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        if self._variant == ModelVariant.BLAZE999_MEDICAL_NER:
            self.sample_text = "45 year old woman diagnosed with CAD"
        elif (
            self._variant == ModelVariant.OPENMED_PII_GERMAN_SUPERCLINICAL_BASE_184M_V1
        ):
            self.sample_text = "Dr. Maria Müller behandelte Patient Hans Schmidt im Universitätsklinikum Berlin am 15.03.2024."
        else:
            self.sample_text = (
                "The patient was administered acetylsalicylic acid for pain relief."
            )
        self.max_length = 128
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeBERTa",
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
        return predicted_tokens_classes
