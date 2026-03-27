# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed model loader implementation for token classification.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing import Optional

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    LLMModelConfig,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available OpenMed model variants for token classification."""

    OPENMED_NER_ANATOMYDETECT_MULTIMED_568M = "NER-AnatomyDetect-MultiMed-568M"
    OPENMED_NER_DNADETECT_MULTIMED_568M = "NER-DNADetect-MultiMed-568M"
    OPENMED_NER_PATHOLOGYDETECT_BIOPATIENT_108M = "NER-PathologyDetect-BioPatient-108M"


class ModelLoader(ForgeModel):
    """OpenMed model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.OPENMED_NER_ANATOMYDETECT_MULTIMED_568M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-AnatomyDetect-MultiMed-568M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_DNADETECT_MULTIMED_568M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-DNADetect-MultiMed-568M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_PATHOLOGYDETECT_BIOPATIENT_108M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-PathologyDetect-BioPatient-108M",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_NER_DNADETECT_MULTIMED_568M

    _VARIANT_SAMPLE_TEXTS = {
        ModelVariant.OPENMED_NER_ANATOMYDETECT_MULTIMED_568M: "The patient complained of pain in the left ventricle region.",
        ModelVariant.OPENMED_NER_DNADETECT_MULTIMED_568M: "The p53 protein plays a crucial role in tumor suppression.",
        ModelVariant.OPENMED_NER_PATHOLOGYDETECT_BIOPATIENT_108M: "Early detection of breast cancer improves survival rates.",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.tokenizer = None
        self.sample_text = self._VARIANT_SAMPLE_TEXTS.get(
            self._variant_name,
            "The p53 protein plays a crucial role in tumor suppression.",
        )

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OpenMed",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load OpenMed model for token classification from Hugging Face."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
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
            max_length=self._variant_config.max_length,
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
