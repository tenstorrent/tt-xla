# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
German-RAG-UAE-Large-V1 model loader implementation for embedding generation.

Uses the BERT-based sentence embedding model fine-tuned from WhereIsAI/UAE-Large-V1
on German RAG embedding triples for semantic similarity and retrieval tasks.
"""
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available German-RAG-UAE-Large-V1 model variants for embedding generation."""

    GERMAN_RAG_UAE_LARGE_V1 = "German-RAG-UAE-LARGE-V1-TRIPLES-MERGED-HESSIAN-AI"


class ModelLoader(ForgeModel):
    """German-RAG-UAE-Large-V1 model loader implementation for embedding generation."""

    _VARIANTS = {
        ModelVariant.GERMAN_RAG_UAE_LARGE_V1: LLMModelConfig(
            pretrained_model_name="avemio/German-RAG-UAE-LARGE-V1-TRIPLES-MERGED-HESSIAN-AI",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GERMAN_RAG_UAE_LARGE_V1

    sample_sentences = [
        "Das Wetter ist heute großartig und perfekt für einen Spaziergang im Park."
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="German-RAG-UAE-Large-V1",
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

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = getattr(self._variant_config, "max_length", 512)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, (tuple, list)):
            last_hidden_state = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs

        # CLS token pooling
        return last_hidden_state[:, 0, :]

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if isinstance(fwd_output, (tuple, list)):
            for t in fwd_output:
                if isinstance(t, torch.Tensor):
                    tensors.append(t.flatten())
        elif hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
            if (
                hasattr(fwd_output, "pooler_output")
                and fwd_output.pooler_output is not None
            ):
                tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
