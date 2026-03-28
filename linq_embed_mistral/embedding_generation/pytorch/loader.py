# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Linq-Embed-Mistral model loader implementation for sentence embedding generation.
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
    """Available Linq-Embed-Mistral model variants for embedding generation."""

    LINQ_EMBED_MISTRAL = "linq-embed-mistral"


class ModelLoader(ForgeModel):
    """Linq-Embed-Mistral model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.LINQ_EMBED_MISTRAL: ModelConfig(
            pretrained_model_name="Linq-AI-Research/Linq-Embed-Mistral",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LINQ_EMBED_MISTRAL

    sample_sentences = [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of France?",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Linq-Embed-Mistral",
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

        inputs = self.tokenizer(
            self.sample_sentences,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        # Last-token pooling: use the last non-padding token's embedding
        if inputs is not None and "attention_mask" in inputs:
            sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
            last_token_embeddings = token_embeddings[
                torch.arange(token_embeddings.shape[0], device=token_embeddings.device),
                sequence_lengths,
            ]
        else:
            last_token_embeddings = token_embeddings[:, -1]

        last_token_embeddings = torch.nn.functional.normalize(
            last_token_embeddings, p=2, dim=1
        )

        return last_token_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "last_hidden_state"):
            return fwd_output.last_hidden_state

        return fwd_output
