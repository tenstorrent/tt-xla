# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CamemBERT model loader implementation for token classification (NER).
DistilCamemBERT is a distilled version of CamemBERT, a French RoBERTa-based model,
fine-tuned for Named Entity Recognition on French text.
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
    """Available CamemBERT token classification model variants."""

    CMARKEA_DISTILCAMEMBERT_BASE_NER = "cmarkea_distilcamembert-base-ner"


class ModelLoader(ForgeModel):
    """CamemBERT model loader for token classification (NER)."""

    _VARIANTS = {
        ModelVariant.CMARKEA_DISTILCAMEMBERT_BASE_NER: LLMModelConfig(
            pretrained_model_name="cmarkea/distilcamembert-base-ner",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CMARKEA_DISTILCAMEMBERT_BASE_NER

    _SAMPLE_TEXTS = {
        ModelVariant.CMARKEA_DISTILCAMEMBERT_BASE_NER: "Le Crédit Mutuel Arkéa est une banque Française, elle est basée à Brest.",
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.text = self._SAMPLE_TEXTS.get(
            self._variant,
            "Le Crédit Mutuel Arkéa est une banque Française, elle est basée à Brest.",
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="CamemBERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load CamemBERT model for token classification from Hugging Face."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for CamemBERT token classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.text,
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

        print(f"Context: {self.text}")
        print(f"Answer: {predicted_tokens_classes}")
