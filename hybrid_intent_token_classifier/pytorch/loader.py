# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hybrid Intent Token Classifier model loader implementation for token classification.

This model is a custom DistilBERT-based architecture from Danswer with dual
classification heads: one for intent classification (sequence-level) and one
for keyword extraction (token-level).
"""

import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
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


class HybridIntentTokenClassifier(nn.Module):
    """Custom DistilBERT model with dual classification heads for intent and keyword extraction."""

    def __init__(self, config):
        super().__init__()
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.intent_classifier = nn.Linear(config.hidden_size, 2)
        self.keyword_classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def forward(self, input_ids, attention_mask):
        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        keyword_logits = self.keyword_classifier(hidden_state)
        return intent_logits, keyword_logits


class ModelVariant(StrEnum):
    """Available model variants for hybrid intent token classification."""

    DANSWER_HYBRID_INTENT_TOKEN_CLASSIFIER = "Danswer/hybrid-intent-token-classifier"


class ModelLoader(ForgeModel):
    """Hybrid Intent Token Classifier model loader implementation."""

    _VARIANTS = {
        ModelVariant.DANSWER_HYBRID_INTENT_TOKEN_CLASSIFIER: LLMModelConfig(
            pretrained_model_name="Danswer/hybrid-intent-token-classifier",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DANSWER_HYBRID_INTENT_TOKEN_CLASSIFIER

    def __init__(self, variant=None):
        super().__init__(variant)
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.max_length = 128
        self.tokenizer = None
        self.sample_text = "What is the weather like in San Francisco today?"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="HybridIntentTokenClassifier",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        model = HybridIntentTokenClassifier(config)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/Danswer/hybrid-intent-token-classifier/resolve/main/pytorch_model.bin",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state_dict)
        self.model = model
        model.eval()
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
        intent_logits, keyword_logits = co_out[0], co_out[1]

        intent_pred = intent_logits.argmax(-1).item()
        intent_labels = {0: "not_intent", 1: "intent"}
        print(f"Input: {self.sample_text}")
        print(f"Intent: {intent_labels.get(intent_pred, intent_pred)}")

        keyword_preds = keyword_logits.argmax(-1)[0]
        inputs = self.load_inputs()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keywords = [
            token
            for token, pred, mask in zip(
                tokens, keyword_preds, inputs["attention_mask"][0]
            )
            if pred == 1 and mask == 1
        ]
        print(f"Keywords: {keywords}")
