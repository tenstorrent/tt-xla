# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed NER model loader implementation for token classification.
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
    """Available OpenMed NER model variants for token classification."""

    OPENMED_NER_BLOODCANCERDETECT_BIOPATIENT_108M = (
        "OpenMed/OpenMed-NER-BloodCancerDetect-BioPatient-108M"
    )
    OPENMED_NER_ONCOLOGYDETECT_EUROMED_212M = (
        "OpenMed/OpenMed-NER-OncologyDetect-EuroMed-212M"
    )
    OPENMED_NER_SPECIESDETECT_SUPERMEDICAL_355M = (
        "OpenMed/OpenMed-NER-SpeciesDetect-SuperMedical-355M"
    )
    OPENMED_NER_DISEASEDETECT_BIGMED_560M = (
        "OpenMed/OpenMed-NER-DiseaseDetect-BigMed-560M"
    )


class ModelLoader(ForgeModel):
    """OpenMed NER model loader implementation for token classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.OPENMED_NER_BLOODCANCERDETECT_BIOPATIENT_108M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-BloodCancerDetect-BioPatient-108M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_ONCOLOGYDETECT_EUROMED_212M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-OncologyDetect-EuroMed-212M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_SPECIESDETECT_SUPERMEDICAL_355M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-SpeciesDetect-SuperMedical-355M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_DISEASEDETECT_BIGMED_560M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-DiseaseDetect-BigMed-560M",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OPENMED_NER_BLOODCANCERDETECT_BIOPATIENT_108M

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.sample_text = (
            "The patient presented with chronic lymphocytic leukemia symptoms."
        )
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="OpenMed NER",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load OpenMed NER model for token classification from Hugging Face."""

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Load pre-trained model from HuggingFace
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
        """Prepare sample input for OpenMed NER token classification."""
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
