# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed model loader implementation for token classification.
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
    """Available OpenMed model variants for token classification."""

    OPENMED_NER_ONCOLOGYDETECT_TINYMED_82M = (
        "OpenMed/OpenMed-NER-OncologyDetect-TinyMed-82M"
    )
    OPENMED_NER_SPECIESDETECT_SUPERCLINICAL_141M = (
        "OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-141M"
    )
    OPENMED_NER_SPECIESDETECT_SUPERCLINICAL_434M = (
        "OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-434M"
    )
    OPENMED_NER_BLOODCANCERDETECT_TINYMED_82M = (
        "OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-82M"
    )
    OPENMED_NER_DISEASEDETECT_BIOCLINICAL_108M = (
        "OpenMed/OpenMed-NER-DiseaseDetect-BioClinical-108M"
    )
    OPENMED_PII_CLINICALE5_SMALL_33M_V1 = "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1"
    OPENMED_PII_ITALIAN_BIOCLINICALMODERN_LARGE_395M_V1 = (
        "OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1"
    )
    OPENMED_PII_EUROMED_LARGE_210M_V1 = "OpenMed/OpenMed-PII-EuroMed-Large-210M-v1"
    OPENMED_PII_SUPERMEDICAL_LARGE_355M_V1 = (
        "OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1"
    )


class ModelLoader(ForgeModel):
    """OpenMed model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.OPENMED_NER_ONCOLOGYDETECT_TINYMED_82M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-OncologyDetect-TinyMed-82M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_SPECIESDETECT_SUPERCLINICAL_141M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-141M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_SPECIESDETECT_SUPERCLINICAL_434M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-434M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_BLOODCANCERDETECT_TINYMED_82M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-82M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_DISEASEDETECT_BIOCLINICAL_108M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-DiseaseDetect-BioClinical-108M",
            max_length=128,
        ),
        ModelVariant.OPENMED_PII_CLINICALE5_SMALL_33M_V1: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
            max_length=384,
        ),
        ModelVariant.OPENMED_PII_ITALIAN_BIOCLINICALMODERN_LARGE_395M_V1: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1",
            max_length=512,
        ),
        ModelVariant.OPENMED_PII_EUROMED_LARGE_210M_V1: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-EuroMed-Large-210M-v1",
            max_length=512,
        ),
        ModelVariant.OPENMED_PII_SUPERMEDICAL_LARGE_355M_V1: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_NER_SPECIESDETECT_SUPERCLINICAL_434M

    _SAMPLE_TEXTS = {
        ModelVariant.OPENMED_NER_ONCOLOGYDETECT_TINYMED_82M: "Mutations in KRAS gene drive oncogenic transformation.",
        ModelVariant.OPENMED_NER_SPECIESDETECT_SUPERCLINICAL_141M: "Escherichia coli bacteria were found in the water samples.",
        ModelVariant.OPENMED_NER_SPECIESDETECT_SUPERCLINICAL_434M: "Escherichia coli and Staphylococcus aureus were isolated from the patient samples.",
        ModelVariant.OPENMED_NER_BLOODCANCERDETECT_TINYMED_82M: "The patient presented with chronic lymphocytic leukemia symptoms.",
        ModelVariant.OPENMED_NER_DISEASEDETECT_BIOCLINICAL_108M: "The patient was diagnosed with diabetes mellitus type 2 and hypertension.",
        ModelVariant.OPENMED_PII_CLINICALE5_SMALL_33M_V1: "Patient John Smith (DOB: 03/15/1985, SSN: 123-45-6789) was seen today.",
        ModelVariant.OPENMED_PII_ITALIAN_BIOCLINICALMODERN_LARGE_395M_V1: "Paziente Marco Bianchi (nato il 15/03/1985, CF: BNCMRC85C15H501Z) è stato visitato oggi.",
        ModelVariant.OPENMED_PII_EUROMED_LARGE_210M_V1: "Dr. Emily Johnson (DOB: 07/22/1990, SSN: 987-65-4321) was admitted to General Hospital on 03/15/2024.",
        ModelVariant.OPENMED_PII_SUPERMEDICAL_LARGE_355M_V1: "Patient Sarah Williams (MRN: 456789, DOB: 11/08/1978) was referred by Dr. Robert Chen for evaluation.",
    }

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = self._SAMPLE_TEXTS[self._variant]
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

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
        """Load OpenMed model for token classification from Hugging Face."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, trust_remote_code=True, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for OpenMed token classification."""
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
        """Decode the model output for token classification."""
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
