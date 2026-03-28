# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""fasttext-ja-vectors model loader implementation for Japanese word embedding generation."""

from typing import Optional

import fasttext
import torch
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
    """Available fasttext-ja-vectors model variants for embedding generation."""

    FASTTEXT_JA_VECTORS = "fasttext-ja-vectors"


class FastTextEmbeddingModule(torch.nn.Module):
    """PyTorch module wrapping pre-computed fastText word embeddings.

    The fastText model itself is not a PyTorch module, so this wrapper
    accepts pre-computed embedding tensors and passes them through.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        return word_embeddings


class ModelLoader(ForgeModel):
    """fasttext-ja-vectors model loader implementation for Japanese word embedding generation."""

    _VARIANTS = {
        ModelVariant.FASTTEXT_JA_VECTORS: ModelConfig(
            pretrained_model_name="facebook/fasttext-ja-vectors",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FASTTEXT_JA_VECTORS

    sample_words = ["東京", "日本語", "人工知能"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._fasttext_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="fasttext-ja-vectors",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_fasttext_model(self):
        model_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.bin",
        )
        self._fasttext_model = fasttext.load_model(model_path)
        return self._fasttext_model

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._fasttext_model is None:
            self._load_fasttext_model()

        embedding_dim = self._fasttext_model.get_dimension()
        model = FastTextEmbeddingModule(embedding_dim)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self._fasttext_model is None:
            self._load_fasttext_model()

        embeddings = []
        for word in self.sample_words:
            vec = self._fasttext_model.get_word_vector(word)
            embeddings.append(torch.tensor(vec))

        inputs = torch.stack(embeddings)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return {"word_embeddings": inputs}
