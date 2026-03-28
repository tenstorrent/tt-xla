# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina Embeddings v3 separation distilled model loader for sentence embedding generation.

This model is a static embedding model (model2vec) distilled from jina-embeddings-v3
with the separation task LoRA applied. It uses pre-computed token embeddings with
mean pooling instead of a transformer encoder.
"""
import torch
import torch.nn.functional as F
from model2vec import StaticModel
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


class StaticEmbeddingModule(torch.nn.Module):
    """PyTorch nn.Module wrapper for model2vec static embeddings.

    Implements embedding lookup with mean pooling and optional normalization,
    compatible with the forge compilation pipeline.
    """

    def __init__(self, embedding_weights, normalize=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            embedding_weights.shape[0], embedding_weights.shape[1]
        )
        self.embedding.weight = torch.nn.Parameter(embedding_weights)
        self.do_normalize = normalize

    def forward(self, input_ids, attention_mask=None, **kwargs):
        token_embeds = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = token_embeds.mean(dim=1)
        if self.do_normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled


class ModelVariant(StrEnum):
    """Available Jina Embeddings v3 separation distilled model variants."""

    JINA_EMBEDDINGS_V3_SEPARATION_DISTILLED = "jina-embeddings-v3-separation-distilled"


class ModelLoader(ForgeModel):
    """Jina Embeddings v3 separation distilled model loader for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.JINA_EMBEDDINGS_V3_SEPARATION_DISTILLED: ModelConfig(
            pretrained_model_name="CISCai/jina-embeddings-v3-separation-distilled",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JINA_EMBEDDINGS_V3_SEPARATION_DISTILLED

    sample_sentences = [
        "Jina Embeddings v3 separation distilled is a static embedding model for text separation tasks"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Jina-Embeddings-v3-Separation-Distilled",
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

        static_model = StaticModel.from_pretrained(pretrained_model_name)
        embedding_weights = torch.tensor(
            static_model.embedding, dtype=dtype_override or torch.float32
        )
        normalize = getattr(static_model, "normalize", False)

        model = StaticEmbeddingModule(embedding_weights, normalize=normalize)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def output_postprocess(self, output, inputs=None):
        return output

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output.flatten()
        return fwd_output
