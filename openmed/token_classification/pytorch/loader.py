# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed model loader implementation for token classification (NER).
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
    """Available OpenMed token classification model variants."""

    OPENMED_NER_BLOODCANCERDETECT_MULTIMED_568M = (
        "OpenMed-NER-BloodCancerDetect-MultiMed-568M"
    )
    OPENMED_NER_DNADETECT_ELECTRAMED_335M = "OpenMed-NER-DNADetect-ElectraMed-335M"
    OPENMED_NER_DNADETECT_TINYMED_135M = "OpenMed-NER-DNADetect-TinyMed-135M"
    OPENMED_NER_GENOMICDETECT_TINYMED_66M = "OpenMed-NER-GenomicDetect-TinyMed-66M"
    OPENMED_NER_PHARMADETECT_TINYMED_66M = "OpenMed-NER-PharmaDetect-TinyMed-66M"
    OPENMED_NER_PHARMADETECT_TINYMED_135M = "OpenMed-NER-PharmaDetect-TinyMed-135M"
    OPENMED_PII_FRENCH_BIOMEDBERT_LARGE_340M_V1 = (
        "OpenMed-PII-French-BiomedBERT-Large-340M-v1"
    )
    OPENMED_PII_GERMAN_CLINICDISCHARGE_BASE_110M_V1 = (
        "OpenMed-PII-German-ClinicDischarge-Base-110M-v1"
    )
    OPENMED_PII_MCLINICALE5_LARGE_560M_V1 = "OpenMed-PII-mClinicalE5-Large-560M-v1"
    OPENMED_PII_SUPERMEDICAL_BASE_125M_V1 = "OpenMed-PII-SuperMedical-Base-125M-v1"


class ModelLoader(ForgeModel):
    """OpenMed model loader implementation for token classification tasks."""

    _VARIANTS = {
        ModelVariant.OPENMED_NER_BLOODCANCERDETECT_MULTIMED_568M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-BloodCancerDetect-MultiMed-568M",
        ),
        ModelVariant.OPENMED_NER_DNADETECT_ELECTRAMED_335M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-DNADetect-ElectraMed-335M",
        ),
        ModelVariant.OPENMED_NER_DNADETECT_TINYMED_135M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-DNADetect-TinyMed-135M",
        ),
        ModelVariant.OPENMED_NER_GENOMICDETECT_TINYMED_66M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-GenomicDetect-TinyMed-66M",
        ),
        ModelVariant.OPENMED_NER_PHARMADETECT_TINYMED_66M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-PharmaDetect-TinyMed-66M",
        ),
        ModelVariant.OPENMED_NER_PHARMADETECT_TINYMED_135M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-PharmaDetect-TinyMed-135M",
        ),
        ModelVariant.OPENMED_PII_FRENCH_BIOMEDBERT_LARGE_340M_V1: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1",
        ),
        ModelVariant.OPENMED_PII_GERMAN_CLINICDISCHARGE_BASE_110M_V1: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1",
        ),
        ModelVariant.OPENMED_PII_MCLINICALE5_LARGE_560M_V1: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1",
        ),
        ModelVariant.OPENMED_PII_SUPERMEDICAL_BASE_125M_V1: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_NER_BLOODCANCERDETECT_MULTIMED_568M

    _VARIANT_SAMPLE_TEXTS = {
        ModelVariant.OPENMED_NER_BLOODCANCERDETECT_MULTIMED_568M: (
            "The patient presented with chronic lymphocytic leukemia symptoms."
        ),
        ModelVariant.OPENMED_NER_DNADETECT_ELECTRAMED_335M: (
            "The p53 protein binds to the BRCA1 gene promoter region in T cells."
        ),
        ModelVariant.OPENMED_NER_DNADETECT_TINYMED_135M: (
            "The p53 protein plays a crucial role in tumor suppression."
        ),
        ModelVariant.OPENMED_NER_GENOMICDETECT_TINYMED_66M: (
            "Mutations in the BRCA1 and TP53 genes were identified in the tumor sample."
        ),
        ModelVariant.OPENMED_NER_PHARMADETECT_TINYMED_66M: (
            "Administration of metformin reduced glucose levels significantly."
        ),
        ModelVariant.OPENMED_NER_PHARMADETECT_TINYMED_135M: (
            "Administration of metformin reduced glucose levels significantly."
        ),
        ModelVariant.OPENMED_PII_FRENCH_BIOMEDBERT_LARGE_340M_V1: (
            "Le patient Jean Dupont, né le 15 mars 1980, réside au 12 rue de la Paix, Paris."
        ),
        ModelVariant.OPENMED_PII_GERMAN_CLINICDISCHARGE_BASE_110M_V1: (
            "Patient Hans Schmidt (geboren am 15.03.1985, SVN: 12 150385 M 234) wurde heute untersucht."
        ),
        ModelVariant.OPENMED_PII_MCLINICALE5_LARGE_560M_V1: (
            "Dr. Sarah Johnson (SSN: 123-45-6789) can be reached at sarah.johnson@hospital.org or 555-123-4567."
        ),
        ModelVariant.OPENMED_PII_SUPERMEDICAL_BASE_125M_V1: (
            "Dr. Sarah Johnson (SSN: 123-45-6789) can be reached at sarah.johnson@hospital.org or 555-123-4567."
        ),
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.sample_text = self._VARIANT_SAMPLE_TEXTS.get(
            self._variant_name,
            "The patient presented with chronic lymphocytic leukemia symptoms.",
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
