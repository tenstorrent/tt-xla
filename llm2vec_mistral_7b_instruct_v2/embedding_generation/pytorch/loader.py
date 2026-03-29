# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLM2Vec-Mistral-7B-Instruct-v2 model loader implementation for embedding generation.
"""
import torch
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
    """Available LLM2Vec-Mistral-7B-Instruct-v2 model variants for embedding generation."""

    LLM2VEC_MISTRAL_7B_INSTRUCT_V2 = "llm2vec-mistral-7b-instruct-v2"


class ModelLoader(ForgeModel):
    """LLM2Vec-Mistral-7B-Instruct-v2 model loader implementation for embedding generation."""

    _VARIANTS = {
        ModelVariant.LLM2VEC_MISTRAL_7B_INSTRUCT_V2: ModelConfig(
            pretrained_model_name="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLM2VEC_MISTRAL_7B_INSTRUCT_V2

    sample_sentences = [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LLM2Vec-Mistral-7B-Instruct-v2",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False, "trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
