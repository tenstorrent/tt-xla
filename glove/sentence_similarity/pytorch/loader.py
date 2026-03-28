# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GloVe-6B-quantized model loader for sentence similarity
using product-quantized static word embeddings.
"""
import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from typing import Optional

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class GloVePQEmbedding(nn.Module):
    """PyTorch module wrapping product-quantized GloVe embeddings.

    Reconstructs 300-dim embeddings from 10 subspace codebooks (each 256x30).
    """

    def __init__(self, codewords, vectors):
        super().__init__()
        self.register_buffer("codewords", codewords)
        self.register_buffer("vectors", vectors.long())

    def forward(self, input_ids):
        codes = self.vectors[input_ids]
        parts = []
        for i in range(codes.shape[-1]):
            parts.append(self.codewords[i][codes[..., i]])
        return torch.cat(parts, dim=-1)


class ModelVariant(StrEnum):
    """Available GloVe model variants."""

    NEUML_GLOVE_6B_QUANTIZED = "NeuML/glove-6B-quantized"


class ModelLoader(ForgeModel):
    """GloVe-6B-quantized model loader."""

    _VARIANTS = {
        ModelVariant.NEUML_GLOVE_6B_QUANTIZED: ModelConfig(
            pretrained_model_name="NeuML/glove-6B-quantized",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEUML_GLOVE_6B_QUANTIZED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.vocab = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="glove-6B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vocab(self):
        if self.vocab is None:
            model_name = self._variant_config.pretrained_model_name
            vocab_path = hf_hub_download(model_name, "vocab.json")
            with open(vocab_path) as f:
                data = json.load(f)
            self.vocab = data["tokens"]
        return self.vocab

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(model_name, "model.safetensors")

        with safe_open(model_path, framework="pt") as f:
            codewords = f.get_tensor("codewords")
            vectors = f.get_tensor("vectors")

        if dtype_override is not None:
            codewords = codewords.to(dtype_override)

        model = GloVePQEmbedding(codewords, vectors)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        vocab = self._load_vocab()

        words = ["the", "quick", "brown", "fox", "jumps"]
        input_ids = [vocab.get(w, 0) for w in words]
        inputs = {"input_ids": torch.tensor([input_ids], dtype=torch.long)}

        return inputs
