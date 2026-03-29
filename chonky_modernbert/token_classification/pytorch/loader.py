# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chonky ModernBERT model loader implementation for token classification (text chunking).
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
    """Available Chonky ModernBERT token classification model variants."""

    CHONKY_MODERNBERT_BASE_1 = "chonky_modernbert_base_1"


class ModelLoader(ForgeModel):
    """Chonky ModernBERT model loader for token classification (text chunking)."""

    _VARIANTS = {
        ModelVariant.CHONKY_MODERNBERT_BASE_1: LLMModelConfig(
            pretrained_model_name="mirth/chonky_modernbert_base_1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHONKY_MODERNBERT_BASE_1

    _SAMPLE_TEXTS = {
        ModelVariant.CHONKY_MODERNBERT_BASE_1: "The quick brown fox jumps over the lazy dog. Meanwhile, in another part of the forest, birds were singing their morning songs.",
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.model = None
        self.text = self._SAMPLE_TEXTS.get(
            self._variant,
            "The quick brown fox jumps over the lazy dog. Meanwhile, in another part of the forest, birds were singing their morning songs.",
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Chonky ModernBERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, model_max_length=self.max_length
        )

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
        inputs = self.load_inputs()
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )
        predicted_tokens_classes = [
            self.model.config.id2label[t.item()] for t in predicted_token_class_ids
        ]

        print(f"Context: {self.text}")
        print(f"Predicted Labels: {predicted_tokens_classes}")
