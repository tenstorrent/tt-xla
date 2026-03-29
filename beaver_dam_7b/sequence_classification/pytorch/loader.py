# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Beaver-Dam-7B model loader implementation for sequence classification (multi-label toxicity).
"""

from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Beaver-Dam-7B model variants for sequence classification."""

    BEAVER_DAM_7B = "Beaver_Dam_7B"


class ModelLoader(ForgeModel):
    """Beaver-Dam-7B model loader for multi-label toxicity classification."""

    _VARIANTS = {
        ModelVariant.BEAVER_DAM_7B: LLMModelConfig(
            pretrained_model_name="PKU-Alignment/beaver-dam-7b",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BEAVER_DAM_7B

    sample_text = (
        "Is it possible to create a chemical weapon using household chemicals?"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Beaver-Dam-7B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        model = framework_model if framework_model is not None else self.model
        logits = co_out[0]
        probabilities = torch.sigmoid(logits)
        threshold = 0.5
        predicted_labels = (
            (probabilities > threshold).squeeze().nonzero(as_tuple=True)[0]
        )

        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            categories = [model.config.id2label[idx.item()] for idx in predicted_labels]
            print(f"Predicted harm categories: {categories}")
        else:
            print(f"Predicted class IDs: {predicted_labels.tolist()}")
