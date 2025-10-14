# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 model loader implementation for embedding tasks.
"""
import torch
from transformers import AutoModel, AutoTokenizer
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
from .src.utils import last_token_pool, get_detailed_instruct

import torch.nn.functional as F


class ModelVariant(StrEnum):
    """Available Qwen 3 model variants for embedding tasks."""

    QWEN_3_EMBEDDING_0_6B = "embedding_0_6b"
    QWEN_3_EMBEDDING_4B = "embedding_4b"
    QWEN_3_EMBEDDING_8B = "embedding_8b"


class ModelLoader(ForgeModel):
    """Qwen 3 model loader implementation for embedding tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_EMBEDDING_0_6B: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-Embedding-0.6B",
        ),
        ModelVariant.QWEN_3_EMBEDDING_4B: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-Embedding-4B",
        ),
        ModelVariant.QWEN_3_EMBEDDING_8B: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-Embedding-8B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_EMBEDDING_0_6B

    # Shared configuration parameters
    sample_task = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    sample_queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    sample_documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="qwen_3_embedding",
            variant=variant,
            group=ModelGroup.GENERALITY
            if variant == ModelVariant.QWEN_3_EMBEDDING_0_6B
            else ModelGroup.RED,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Qwen 3 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen 3 model instance for embedding tasks.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["use_cache"] = False

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, max_length=128):
        """Load and return sample inputs for the Qwen 3 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            max_length: Maximum sequence length for tokenization.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Prepare input texts
        queries = [
            get_detailed_instruct(self.sample_task, query)
            for query in self.sample_queries
        ]
        input_texts = queries + self.sample_documents

        # Tokenize the input texts
        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to process model outputs for embedding similarity.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            similarity_scores : Pairwise similarity scores between queries and documents
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        # Get embeddings using last token pooling
        embeddings = last_token_pool(outputs, inputs["attention_mask"])

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity scores between queries and documents
        num_queries = len(self.sample_queries)
        scores = embeddings[:num_queries] @ embeddings[num_queries:].T

        return scores.tolist()
