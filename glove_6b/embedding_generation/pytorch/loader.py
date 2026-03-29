# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GloVe-6B model loader implementation for word embedding generation.

Uses the NeuML/glove-6B StaticVectors model which provides 300-dimensional
GloVe word embeddings trained on 6 billion tokens.
"""
import numpy as np
import torch
import torch.nn as nn
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
    """Available GloVe-6B model variants for embedding generation."""

    GLOVE_6B = "NeuML/glove-6B"


class GloVeEmbeddingModel(nn.Module):
    """PyTorch wrapper around StaticVectors GloVe embeddings."""

    def __init__(self, model_name: str):
        super().__init__()
        from staticvectors import StaticVectors

        self._sv = StaticVectors(model_name)
        embedding_dim = 300
        self._embedding_dim = embedding_dim

    def forward(self, input_texts: list[str]) -> torch.Tensor:
        embeddings = self._sv.embeddings(input_texts)
        return torch.tensor(np.array(embeddings), dtype=torch.float32)


class ModelLoader(ForgeModel):
    """GloVe-6B model loader implementation for word embedding generation."""

    _VARIANTS = {
        ModelVariant.GLOVE_6B: ModelConfig(
            pretrained_model_name="NeuML/glove-6B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLOVE_6B

    sample_sentences = ["This is an example sentence for generating word embeddings"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GloVe-6B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model = GloVeEmbeddingModel(self._variant_config.pretrained_model_name)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        return self.sample_sentences
