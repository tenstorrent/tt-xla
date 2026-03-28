# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed model loader implementation for token classification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
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

    NER_PHARMADETECT_MODERNMED_149M = "OpenMed/OpenMed-NER-PharmaDetect-ModernMed-149M"
    NER_GENOMICDETECT_BIOMED_335M = "OpenMed/OpenMed-NER-GenomicDetect-BioMed-335M"
    NER_PROTEINDETECT_PUBMED_V2_109M = (
        "OpenMed/OpenMed-NER-ProteinDetect-PubMed-v2-109M"
    )
    NER_ORGANISMDETECT_MODERNCLINICAL_149M = (
        "OpenMed/OpenMed-NER-OrganismDetect-ModernClinical-149M"
    )
    NER_GENOMEDETECT_BIOMED_109M = "OpenMed/OpenMed-NER-GenomeDetect-BioMed-109M"
    NER_ANATOMYDETECT_MODERNMED_395M = (
        "OpenMed/OpenMed-NER-AnatomyDetect-ModernMed-395M"
    )
    PII_ITALIAN_BIOCLINICALMODERN_BASE_149M = (
        "OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1"
    )
    PII_FRENCH_BIOCLINICALMODERN_BASE_149M = (
        "OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1"
    )
    PII_CLINICDISCHARGE_BASE_110M = "OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1"
    PII_SPANISH_CLINICALBGE_LARGE_568M = (
        "OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1"
    )


_VARIANT_SAMPLE_TEXTS = {
    ModelVariant.NER_PHARMADETECT_MODERNMED_149M: (
        "Administration of metformin reduced glucose levels significantly."
    ),
    ModelVariant.NER_GENOMICDETECT_BIOMED_335M: (
        "The HeLa cell line was used to study BRCA1 gene expression patterns."
    ),
    ModelVariant.NER_GENOMEDETECT_BIOMED_109M: (
        "Mutations in the TP53 gene are commonly associated with various cancers."
    ),
    ModelVariant.NER_PROTEINDETECT_PUBMED_V2_109M: (
        "Casein micelles are the primary protein component of milk."
    ),
    ModelVariant.NER_ORGANISMDETECT_MODERNCLINICAL_149M: (
        "Caenorhabditis elegans is a model organism for genetic studies."
    ),
    ModelVariant.NER_ANATOMYDETECT_MODERNMED_395M: (
        "The patient complained of pain in the left ventricle region."
    ),
    ModelVariant.PII_ITALIAN_BIOCLINICALMODERN_BASE_149M: (
        "Il paziente Mario Rossi, nato il 15 marzo 1980, è stato ricoverato presso l'Ospedale San Raffaele di Milano."
    ),
    ModelVariant.PII_FRENCH_BIOCLINICALMODERN_BASE_149M: (
        "Le patient Jean Dupont, né le 12 avril 1975, a été admis à l'Hôpital Pitié-Salpêtrière de Paris."
    ),
    ModelVariant.PII_CLINICDISCHARGE_BASE_110M: (
        "Dr. John Smith treated patient Jane Doe at Massachusetts General Hospital on 03/15/2024."
    ),
    ModelVariant.PII_SPANISH_CLINICALBGE_LARGE_568M: (
        "El paciente Carlos Garcia, nacido el 20 de enero de 1985, fue atendido en el Hospital Clinico de Barcelona."
    ),
}


class ModelLoader(ForgeModel):
    """OpenMed model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.NER_PHARMADETECT_MODERNMED_149M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-PharmaDetect-ModernMed-149M",
            max_length=128,
        ),
        ModelVariant.NER_GENOMICDETECT_BIOMED_335M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-GenomicDetect-BioMed-335M",
            max_length=128,
        ),
        ModelVariant.NER_PROTEINDETECT_PUBMED_V2_109M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-ProteinDetect-PubMed-v2-109M",
            max_length=128,
        ),
        ModelVariant.NER_ORGANISMDETECT_MODERNCLINICAL_149M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-OrganismDetect-ModernClinical-149M",
            max_length=128,
        ),
        ModelVariant.NER_GENOMEDETECT_BIOMED_109M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-GenomeDetect-BioMed-109M",
            max_length=128,
        ),
        ModelVariant.NER_ANATOMYDETECT_MODERNMED_395M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-AnatomyDetect-ModernMed-395M",
            max_length=128,
        ),
        ModelVariant.PII_ITALIAN_BIOCLINICALMODERN_BASE_149M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1",
            max_length=128,
        ),
        ModelVariant.PII_FRENCH_BIOCLINICALMODERN_BASE_149M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1",
            max_length=128,
        ),
        ModelVariant.PII_CLINICDISCHARGE_BASE_110M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1",
            max_length=128,
        ),
        ModelVariant.PII_SPANISH_CLINICALBGE_LARGE_568M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NER_PHARMADETECT_MODERNMED_149M

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = _VARIANT_SAMPLE_TEXTS.get(
            self._variant_name,
            "Administration of metformin reduced glucose levels significantly.",
        )
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
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
