# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed NER model loader implementation for token classification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available OpenMed NER model variants."""

    ONCOLOGY_DETECT_SUPERMEDICAL_125M = "OncologyDetect-SuperMedical-125M"
    DISEASE_DETECT_MODERNMED_395M = "DiseaseDetect-ModernMed-395M"
    BLOOD_CANCER_DETECT_MODERNMED_395M = "BloodCancerDetect-ModernMed-395M"


_VARIANT_SAMPLE_TEXTS = {
    ModelVariant.ONCOLOGY_DETECT_SUPERMEDICAL_125M: "Mutations in KRAS gene drive oncogenic transformation in colorectal cancer cells.",
    ModelVariant.DISEASE_DETECT_MODERNMED_395M: "The patient was diagnosed with diabetes mellitus type 2 and chronic obstructive pulmonary disease.",
    ModelVariant.BLOOD_CANCER_DETECT_MODERNMED_395M: "The patient presented with chronic lymphocytic leukemia symptoms.",
}


class ModelLoader(ForgeModel):
    """OpenMed NER model loader implementation for token classification tasks."""

    _VARIANTS = {
        ModelVariant.ONCOLOGY_DETECT_SUPERMEDICAL_125M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-OncologyDetect-SuperMedical-125M",
        ),
        ModelVariant.DISEASE_DETECT_MODERNMED_395M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-DiseaseDetect-ModernMed-395M",
        ),
        ModelVariant.BLOOD_CANCER_DETECT_MODERNMED_395M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-BloodCancerDetect-ModernMed-395M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ONCOLOGY_DETECT_SUPERMEDICAL_125M

    def __init__(self, variant=None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.sample_text = _VARIANT_SAMPLE_TEXTS.get(
            self._variant_name,
            "Mutations in KRAS gene drive oncogenic transformation in colorectal cancer cells.",
        )
        self.max_length = 128

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OpenMed",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
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

        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
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
        print(f"Predicted Labels: {predicted_tokens_classes}")
