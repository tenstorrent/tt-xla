# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed NER model loader implementation for biomedical entity recognition.
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
    """Available OpenMed NER model variants."""

    ANATOMY_DETECT_SUPERMEDICAL_355M = "AnatomyDetect-SuperMedical-355M"
    ANATOMY_DETECT_MULTIMED_335M = "AnatomyDetect-MultiMed-335M"
    GENOME_DETECT_BIOPATIENT_108M = "GenomeDetect-BioPatient-108M"
    SPECIES_DETECT_BIOCLINICAL_108M = "SpeciesDetect-BioClinical-108M"
    PROTEIN_DETECT_MODERNCLINICAL_395M = "ProteinDetect-ModernClinical-395M"
    GENOME_DETECT_PUBMED_V2_109M = "GenomeDetect-PubMed-v2-109M"


class ModelLoader(ForgeModel):
    """OpenMed NER model loader implementation for biomedical entity recognition."""

    _VARIANTS = {
        ModelVariant.ANATOMY_DETECT_SUPERMEDICAL_355M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-AnatomyDetect-SuperMedical-355M",
        ),
        ModelVariant.ANATOMY_DETECT_MULTIMED_335M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-AnatomyDetect-MultiMed-335M",
        ),
        ModelVariant.GENOME_DETECT_BIOPATIENT_108M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-GenomeDetect-BioPatient-108M",
        ),
        ModelVariant.SPECIES_DETECT_BIOCLINICAL_108M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-SpeciesDetect-BioClinical-108M",
        ),
        ModelVariant.PROTEIN_DETECT_MODERNCLINICAL_395M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-ProteinDetect-ModernClinical-395M",
        ),
        ModelVariant.GENOME_DETECT_PUBMED_V2_109M: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-GenomeDetect-PubMed-v2-109M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANATOMY_DETECT_SUPERMEDICAL_355M

    _SAMPLE_TEXTS = {
        ModelVariant.ANATOMY_DETECT_SUPERMEDICAL_355M: "The patient complained of pain in the left ventricle region.",
        ModelVariant.ANATOMY_DETECT_MULTIMED_335M: "The patient complained of pain in the left ventricle region.",
        ModelVariant.GENOME_DETECT_BIOPATIENT_108M: "The EGFR gene mutation was identified in lung cancer patients.",
        ModelVariant.SPECIES_DETECT_BIOCLINICAL_108M: "Escherichia coli bacteria were found in the water samples.",
        ModelVariant.PROTEIN_DETECT_MODERNCLINICAL_395M: "The BRCA1 protein interacts with the p53 tumor suppressor in breast cancer cells.",
        ModelVariant.GENOME_DETECT_PUBMED_V2_109M: "The EGFR gene mutation was identified in lung cancer patients.",
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = self._SAMPLE_TEXTS.get(
            self._variant,
            "The patient complained of pain in the left ventricle region.",
        )
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "ner"
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
        print(f"NER Tags: {predicted_tokens_classes}")
