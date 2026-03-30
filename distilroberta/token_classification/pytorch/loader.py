# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilRoBERTa model loader implementation for token classification (NER).
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
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
    """Available DistilRoBERTa model variants for token classification."""

    DISTILROBERTA_BASE_NER_CONLL2003 = "philschmid/distilroberta-base-ner-conll2003"


class ModelLoader(ForgeModel):
    """DistilRoBERTa model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.DISTILROBERTA_BASE_NER_CONLL2003: LLMModelConfig(
            pretrained_model_name="philschmid/distilroberta-base-ner-conll2003",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILROBERTA_BASE_NER_CONLL2003

    _SAMPLE_TEXTS = {
        ModelVariant.DISTILROBERTA_BASE_NER_CONLL2003: "My name is Philipp and I live in Germany",
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = self._SAMPLE_TEXTS.get(
            self._variant,
            "My name is Philipp and I live in Germany",
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="DistilRoBERTa",
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
        model.eval()
        self.model = model
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

    def decode_output(self, co_out, framework_model=None):
        inputs = self.load_inputs()
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )

        model = framework_model if framework_model else getattr(self, "model", None)
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_tokens_classes = [
                model.config.id2label[t.item()] for t in predicted_token_class_ids
            ]
            print(f"Context: {self.sample_text}")
            print(f"NER Tags: {predicted_tokens_classes}")
        else:
            print(f"Context: {self.sample_text}")
            print(f"Predicted token class IDs: {predicted_token_class_ids.tolist()}")
