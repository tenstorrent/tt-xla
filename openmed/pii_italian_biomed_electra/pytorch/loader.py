# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed PII Italian BiomedELECTRA model loader implementation for token classification.
"""

import torch
from typing import Optional
from transformers import AutoModelForTokenClassification, AutoTokenizer

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
    OPENMED_PII_ITALIAN_BIOMED_ELECTRA_LARGE_335M_V1 = (
        "PII-Italian-BiomedELECTRA-Large-335M-v1"
    )


class ModelLoader(ForgeModel):
    """OpenMed PII Italian BiomedELECTRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.OPENMED_PII_ITALIAN_BIOMED_ELECTRA_LARGE_335M_V1: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_PII_ITALIAN_BIOMED_ELECTRA_LARGE_335M_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "Il paziente Mario Rossi è nato il 15/03/1985 e risiede in Via Roma 15, 20121 Milano."
        self.tokenizer = None

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
        """Load the OpenMed PII Italian BiomedELECTRA model."""
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
        """Prepare sample input for PII token classification."""
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
