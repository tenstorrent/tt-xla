# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Reranker sequence classification model loader implementation for passage ranking.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """Available Qwen3-Reranker-seq-cls model variants for passage ranking."""

    QWEN_3_RERANKER_0_6B_SEQ_CLS = "0_6B-seq-cls"


class ModelLoader(ForgeModel):
    """Qwen3-Reranker sequence classification model loader implementation for passage ranking.

    This is a sequence classification conversion of the Qwen3-Reranker model,
    using AutoModelForSequenceClassification instead of the original causal LM approach.
    """

    _VARIANTS = {
        ModelVariant.QWEN_3_RERANKER_0_6B_SEQ_CLS: ModelConfig(
            pretrained_model_name="tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_RERANKER_0_6B_SEQ_CLS

    # System prompt for the reranker
    _SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    )

    # Sample query-passage pairs for testing
    sample_pairs = [
        (
            "Which planet is known as the Red Planet?",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        ),
    ]

    # Default task instruction
    sample_instruction = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen3RerankerSeqCls",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _format_input(self, instruction, query, document):
        """Format a query-document pair using the reranker's chat template."""
        prefix = (
            f"<|im_start|>system\n{self._SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}{suffix}"

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            padding_side="left",
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        pairs = [
            self._format_input(self.sample_instruction, query, document)
            for query, document in self.sample_pairs
        ]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
