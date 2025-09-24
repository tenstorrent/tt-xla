# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE M3 embedding model loader using FlagEmbedding's BGEM3FlagModel.
"""
import torch
import numpy as np
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
    """Available BGE M3 variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """Loader for BGE M3 embedding model."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="BAAI/bge-m3",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="bge_m3_encode",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self):
        from FlagEmbedding import BGEM3FlagModel

        model_name = self._variant_config.pretrained_model_name
        flag_model = BGEM3FlagModel(model_name)
        self.model = flag_model

        return self.model.encode_single_device

    def load_inputs(self):
        sentences = [
            "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
            "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
        ]

        return {
            "sentences": sentences,
            "return_dense": True,
            "return_sparse": True,
            "return_colbert_vecs": True,
        }
