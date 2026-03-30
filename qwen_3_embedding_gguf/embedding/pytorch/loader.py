# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 Embedding GGUF model loader implementation for embedding tasks.
"""
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
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
from .src.utils import last_token_pool, get_detailed_instruct

import torch.nn.functional as F


class ModelVariant(StrEnum):
    """Available Qwen 3 Embedding GGUF model variants."""

    QWEN_3_EMBEDDING_0_6B_Q4_K_M = "Embedding_0_6B_Q4_K_M"


class ModelLoader(ForgeModel):
    """Qwen 3 Embedding GGUF model loader implementation for embedding tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_EMBEDDING_0_6B_Q4_K_M: ModelConfig(
            pretrained_model_name="PeterAM4/Qwen3-Embedding-0.6B-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_EMBEDDING_0_6B_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.QWEN_3_EMBEDDING_0_6B_Q4_K_M: "Qwen3-Embedding-0.6B-Q4_K_M-imat.gguf",
    }

    sample_task = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    sample_queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    sample_documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3 Embedding GGUF",
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
        tokenizer_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._GGUF_FILES[self._variant]
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, max_length=128):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        queries = [
            get_detailed_instruct(self.sample_task, query)
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
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        embeddings = last_token_pool(outputs, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        num_queries = len(self.sample_queries)
        scores = embeddings[:num_queries] @ embeddings[num_queries:].T

        return scores.tolist()

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
