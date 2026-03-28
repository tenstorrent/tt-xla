# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HuSpaCy Hungarian NLP model loader for sentence embedding generation.

HuSpaCy hu_core_news_md is a medium-sized spaCy pipeline for Hungarian NLP
featuring tok2vec, NER, POS tagging, dependency parsing, and lemmatization.
This loader extracts the static word vectors (200k vectors, 100 dimensions)
and wraps them as a PyTorch embedding model for sentence embedding generation.
"""
import torch
import torch.nn as nn
import spacy
from typing import Optional

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


class HuSpacyEmbeddingModel(nn.Module):
    """Wraps HuSpaCy word vectors as a PyTorch module for sentence embedding generation."""

    def __init__(self, nlp):
        super().__init__()
        vectors_np = nlp.vocab.vectors.data.copy()
        vectors_tensor = torch.from_numpy(vectors_np).float()
        # Prepend a zero vector for OOV tokens (index 0), shift real indices by +1
        oov = torch.zeros(1, vectors_tensor.shape[1])
        all_vectors = torch.cat([oov, vectors_tensor], dim=0)
        self.embedding = nn.Embedding.from_pretrained(
            all_vectors, freeze=True, padding_idx=0
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = self.embedding(input_ids)
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        summed = torch.sum(token_embeddings * mask_expanded, dim=1)
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled


class ModelVariant(StrEnum):
    """Available HuSpaCy model variants."""

    HU_CORE_NEWS_MD = "hu_core_news_md"


class ModelLoader(ForgeModel):
    """HuSpaCy model loader for Hungarian sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.HU_CORE_NEWS_MD: LLMModelConfig(
            pretrained_model_name="huspacy/hu_core_news_md",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HU_CORE_NEWS_MD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._nlp = None
        self.sample_text = "Budapest Magyarország fővárosa és egyben legnagyobb városa."

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HuSpaCy",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_nlp(self):
        if self._nlp is None:
            model_name = self._variant_config.pretrained_model_name
            self._nlp = spacy.load(model_name)
        return self._nlp

    def load_model(self, *, dtype_override=None, **kwargs):
        nlp = self._load_nlp()
        model = HuSpacyEmbeddingModel(nlp)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        nlp = self._load_nlp()
        doc = nlp.make_doc(self.sample_text)
        max_length = self._variant_config.max_length

        # Get vector row indices for each token
        input_ids = []
        for token in doc:
            row = nlp.vocab.vectors.find(key=token.orth)
            input_ids.append(row + 1 if row >= 0 else 0)

        # Pad or truncate to max_length
        if len(input_ids) >= max_length:
            input_ids = input_ids[:max_length]
            attention_mask = [1] * max_length
        else:
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            input_ids = input_ids + [0] * (max_length - len(input_ids))

        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output.flatten()
        return fwd_output
