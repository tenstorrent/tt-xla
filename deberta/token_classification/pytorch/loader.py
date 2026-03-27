# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeBERTa model loader implementation for token classification (NER) task.
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
    LLMModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available DeBERTa token classification model variants."""

    OPENMED_NER_GENOMIC_DETECT = "OpenMed_NER_GenomicDetect_SuperClinical_184M"


_VARIANT_SAMPLE_TEXTS = {
    ModelVariant.OPENMED_NER_GENOMIC_DETECT: "The BRCA2 gene is associated with hereditary breast cancer.",
}


class ModelLoader(ForgeModel):
    """DeBERTa model loader implementation for token classification (NER) task."""

    _VARIANTS = {
        ModelVariant.OPENMED_NER_GENOMIC_DETECT: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-GenomicDetect-SuperClinical-184M",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_NER_GENOMIC_DETECT

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.model = None
        self.sample_text = _VARIANT_SAMPLE_TEXTS.get(
            self._variant_name,
            "The BRCA2 gene is associated with hereditary breast cancer.",
        )

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DeBERTa",
            variant=variant,
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

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model.eval()
        return self.model

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
