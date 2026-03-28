# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Reranker Sequence Classification model loader for passage ranking.

This uses the tomaarsen/Qwen3-Reranker-4B-seq-cls model which converts the
original Qwen3-Reranker from a CausalLM to a standard SequenceClassification
architecture, allowing direct logit-based relevance scoring.
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
    """Available Qwen3 Reranker Sequence Classification model variants."""

    QWEN_3_RERANKER_4B_SEQ_CLS = "4B_seq_cls"


class ModelLoader(ForgeModel):
    """Qwen3 Reranker Sequence Classification model loader for passage ranking."""

    _VARIANTS = {
        ModelVariant.QWEN_3_RERANKER_4B_SEQ_CLS: ModelConfig(
            pretrained_model_name="tomaarsen/Qwen3-Reranker-4B-seq-cls",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_RERANKER_4B_SEQ_CLS

    # System prompt for the reranker
    _SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    )

    # Sample query-passage pairs for testing
    sample_pairs = [
        (
            "What is the capital of China?",
            "The capital of China is Beijing.",
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
        user_content = (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Append empty thinking block as per model card
        text += "<think>\n\n</think>\n\n"
        return text

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
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        texts = [
            self._format_input(self.sample_instruction, query, doc)
            for query, doc in self.sample_pairs
        ]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
