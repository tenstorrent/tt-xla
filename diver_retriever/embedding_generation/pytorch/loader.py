# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Diver-Retriever-4B model loader implementation for embedding generation.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
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
    """Available Diver-Retriever model variants for embedding generation."""

    DIVER_RETRIEVER_4B_1020 = "Diver-Retriever-4B-1020"


class ModelLoader(ForgeModel):
    """Diver-Retriever-4B model loader implementation for embedding generation."""

    _VARIANTS = {
        ModelVariant.DIVER_RETRIEVER_4B_1020: ModelConfig(
            pretrained_model_name="AQ-MedAI/Diver-Retriever-4B-1020",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIVER_RETRIEVER_4B_1020

    sample_task = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    sample_queries = [
        "What are the symptoms of diabetes?",
    ]
    sample_documents = [
        "Diabetes symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision.",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Diver-Retriever-4B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            padding_side="left",
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, max_length=128):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        queries = [
            f"Instruct: {self.sample_task}\nQuery:{query}"
            for query in self.sample_queries
        ]
        input_texts = queries + self.sample_documents

        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        last_hidden_state = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )

        # Last token pooling (left-padded, so last token is always at position -1)
        embeddings = last_hidden_state[:, -1]

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity scores between queries and documents
        num_queries = len(self.sample_queries)
        scores = embeddings[:num_queries] @ embeddings[num_queries:].T

        return scores.tolist()
