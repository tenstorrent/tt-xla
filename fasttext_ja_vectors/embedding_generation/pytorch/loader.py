# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
fasttext-ja-vectors model loader implementation for Japanese word embedding generation.
"""
import torch
import fasttext
from huggingface_hub import hf_hub_download
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
    """Available fasttext-ja-vectors model variants for embedding generation."""

    FASTTEXT_JA_VECTORS = "fasttext-ja-vectors"


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

        # Wrap fasttext model in a torch.nn.Module for compatibility
        ft_model = self._fasttext_model
        embedding_dim = ft_model.get_dimension()

        class FastTextWrapper(torch.nn.Module):
            def __init__(self, fasttext_model, dim):
                super().__init__()
                self.fasttext_model = fasttext_model
                self.dim = dim
                # Register a dummy parameter so PyTorch recognizes this as a module
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, word_embeddings):
                # In inference mode, pass through pre-computed embeddings
                return word_embeddings

        model = FastTextWrapper(ft_model, embedding_dim)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self._fasttext_model is None:
            self._load_fasttext_model()

        # Convert sample words to embedding tensors using fasttext
        embeddings = []
        for word in self.sample_words:
            vec = self._fasttext_model.get_word_vector(word)
            embeddings.append(torch.tensor(vec))

        inputs = torch.stack(embeddings)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return {"word_embeddings": inputs}
