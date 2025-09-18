# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from typing import Any, Dict, Sequence
from infra import ComparisonConfig, Model, RunMode, TorchModelTester

# Install FlagEmbedding if not available
try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "FlagEmbedding"])
    from FlagEmbedding import BGEM3FlagModel


class BGE_M3Tester(TorchModelTester):
    """Tester for BGE-M3 model."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        model = BGEM3FlagModel("BAAI/bge-m3")
        self.model = model.encode
        return self.model

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        # texts = [
        #     "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        #     "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
        # ]

        # tokenized = self.model.tokenizer(
        #     texts, return_tensors="pt", padding=True, truncation=True
        # )

        # forward_inputs = {
        #     "text_input": tokenized,
        #     "return_dense": True,
        #     "return_sparse": True,
        #     "return_colbert_vecs": True,
        #     "return_sparse_embedding": False,
        # }
        # return forward_inputs
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
