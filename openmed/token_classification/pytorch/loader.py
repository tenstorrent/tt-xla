# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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


_VARIANT_SAMPLE_TEXTS = {
    ModelVariant.NER_PHARMADETECT_MODERNMED_149M: (
        "Administration of metformin reduced glucose levels significantly."
    ),
    ModelVariant.NER_GENOMICDETECT_BIOMED_335M: (
        "The HeLa cell line was used to study BRCA1 gene expression patterns."
    ),
    ModelVariant.NER_PROTEINDETECT_PUBMED_V2_109M: (
        "Casein micelles are the primary protein component of milk."
    ),
    ModelVariant.NER_ORGANISMDETECT_MODERNCLINICAL_149M: (
        "Caenorhabditis elegans is a model organism for genetic studies."
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
