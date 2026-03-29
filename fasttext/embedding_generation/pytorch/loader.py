# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NeuML/fasttext model loader implementation for word embedding generation."""

import json
from typing import Optional

import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

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
    """Available fasttext model variants for embedding generation."""

    FASTTEXT = "NeuML/fasttext"


class FastTextEmbeddingModule(torch.nn.Module):
    """PyTorch module wrapping pre-computed fastText word embeddings.

    The StaticVectors fastText model is not a PyTorch module, so this wrapper
    accepts pre-computed embedding tensors and passes them through.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        return word_embeddings


class ModelLoader(ForgeModel):
    """NeuML/fasttext model loader implementation for word embedding generation."""

    _VARIANTS = {
        ModelVariant.FASTTEXT: ModelConfig(
            pretrained_model_name="NeuML/fasttext",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FASTTEXT

    sample_words = ["happy", "language", "artificial"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._embeddings = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="fasttext",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_embeddings(self):
        model_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.safetensors",
        )
        self._embeddings = load_file(model_path)
        vocab_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="vocab.json",
        )
        with open(vocab_path) as f:
            self._vocab = json.load(f)

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._embeddings is None:
            self._load_embeddings()

        embedding_tensor = self._embeddings["embeddings"]
        embedding_dim = embedding_tensor.shape[1]
        model = FastTextEmbeddingModule(embedding_dim)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self._embeddings is None:
            self._load_embeddings()

        embedding_tensor = self._embeddings["embeddings"]
        embeddings = []
        for word in self.sample_words:
            if word in self._vocab:
                idx = self._vocab[word]
                embeddings.append(embedding_tensor[idx])
            else:
                embeddings.append(torch.zeros(embedding_tensor.shape[1]))

        inputs = torch.stack(embeddings)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return {"word_embeddings": inputs}
