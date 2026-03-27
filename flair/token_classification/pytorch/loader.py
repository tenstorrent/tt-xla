# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair NER model loader implementation for token classification.

Uses the flair library's SequenceTagger to load the pretrained model, then
extracts the underlying XLM-RoBERTa transformer and classification head into
a standard PyTorch module that accepts tensor inputs.
"""

import torch
import torch.nn as nn
from flair.models import SequenceTagger
from transformers import AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Flair NER model variants."""

    NER_ENGLISH_LARGE = "flair/ner-english-large"


class FlairNERWrapper(nn.Module):
    """Wrapper that extracts transformer + classifier from a flair SequenceTagger.

    Converts the flair-native model into a standard PyTorch module that
    accepts tokenized tensors (input_ids, attention_mask) and returns logits.
    """

    def __init__(self, model_name):
        super().__init__()
        tagger = SequenceTagger.load(model_name)
        self.transformer = tagger.embeddings.model.model
        self.classifier = tagger.linear

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)
        return logits


class ModelLoader(ForgeModel):
    """Flair NER model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.NER_ENGLISH_LARGE: ModelConfig(
            pretrained_model_name="flair/ner-english-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NER_ENGLISH_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.sample_text = "George Washington went to Washington"
        self.max_length = 128

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Flair NER",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

        model = FlairNERWrapper(pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

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

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )

        print(f"Context: {self.sample_text}")
        print(f"Predicted NER Tag IDs: {predicted_token_class_ids.tolist()}")
