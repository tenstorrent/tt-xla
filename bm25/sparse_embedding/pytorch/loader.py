# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qdrant BM25 sparse embedding model loader implementation."""

from typing import Optional

import torch

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
    """Available Qdrant BM25 model variants."""

    BM25 = "bm25"


class Bm25Module(torch.nn.Module):
    """PyTorch module wrapping BM25 term frequency computation.

    Computes the BM25 term frequency score:
        score = (tf * (k + 1)) / (tf + k * (1 - b + b * doc_len / avg_len))

    Inputs are pre-tokenized and hashed by the loader's load_inputs method.
    """

    def __init__(self, k: float = 1.2, b: float = 0.75, avg_len: float = 256.0):
        super().__init__()
        self.register_buffer("k", torch.tensor(k))
        self.register_buffer("b", torch.tensor(b))
        self.register_buffer("avg_len", torch.tensor(avg_len))

    def forward(self, token_frequencies: torch.Tensor, doc_lengths: torch.Tensor):
        """Compute BM25 term frequency scores.

        Args:
            token_frequencies: Tensor of shape (batch, max_tokens) with raw term counts.
            doc_lengths: Tensor of shape (batch, 1) with document lengths.

        Returns:
            Tensor of shape (batch, max_tokens) with BM25 TF scores.
        """
        numerator = token_frequencies * (self.k + 1)
        denominator = token_frequencies + self.k * (
            1 - self.b + self.b * doc_lengths / self.avg_len
        )
        return numerator / denominator


class ModelLoader(ForgeModel):
    """Qdrant BM25 sparse embedding model loader."""

    _VARIANTS = {
        ModelVariant.BM25: ModelConfig(
            pretrained_model_name="Qdrant/bm25",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BM25

    sample_text = [
        "Semantic search using sparse retrieval with learned term weights.",
        "History can only prepare us to be surprised yet again.",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BM25",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BM25 PyTorch module."""
        model = Bm25Module()
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return pre-tokenized sample inputs as tensors.

        Uses fastembed's BM25 tokenizer to preprocess text, then builds
        padded tensors of token frequencies suitable for the Bm25Module forward pass.
        """
        from collections import Counter

        import mmh3
        from fastembed.common.utils import remove_non_alphanumeric
        from fastembed.sparse.bm25 import Bm25
        from fastembed.sparse.utils.tokenizer import SimpleTokenizer
        from py_rust_stemmers import SnowballStemmer

        stemmer = SnowballStemmer("english")

        # Load stopwords from the cached model
        bm25 = Bm25(model_name="Qdrant/bm25")
        stopwords = bm25.stopwords
        punctuation = bm25.punctuation

        all_token_counts = []
        all_token_ids = []

        for text in self.sample_text:
            text = remove_non_alphanumeric(text)
            tokens = SimpleTokenizer.tokenize(text)

            # Stem and filter tokens
            stemmed = []
            for token in tokens:
                lower = token.lower()
                if token in punctuation or lower in stopwords or len(token) > 40:
                    continue
                stemmed_token = stemmer.stem_word(lower)
                if stemmed_token:
                    stemmed.append(stemmed_token)

            counts = Counter(stemmed)
            token_ids = [abs(mmh3.hash(t)) for t in counts.keys()]
            token_freqs = list(counts.values())

            all_token_ids.append(token_ids)
            all_token_counts.append(token_freqs)

        # Pad to same length
        max_tokens = max(len(tc) for tc in all_token_counts)
        padded_freqs = []
        for freqs in all_token_counts:
            padded_freqs.append(freqs + [0] * (max_tokens - len(freqs)))

        dtype = dtype_override or torch.float32
        token_frequencies = torch.tensor(padded_freqs, dtype=dtype)
        doc_lengths = torch.tensor([[len(tc)] for tc in all_token_counts], dtype=dtype)

        return {
            "token_frequencies": token_frequencies,
            "doc_lengths": doc_lengths,
        }
