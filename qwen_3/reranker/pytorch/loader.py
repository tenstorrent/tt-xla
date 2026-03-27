# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 Reranker model loader implementation for passage ranking.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Available Qwen 3 Reranker model variants for passage ranking."""

    QWEN_3_RERANKER_0_6B = "0_6B"
    QWEN_3_RERANKER_4B = "4B"


class ModelLoader(ForgeModel):
    """Qwen 3 Reranker model loader implementation for passage ranking."""

    _VARIANTS = {
        ModelVariant.QWEN_3_RERANKER_0_6B: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-Reranker-0.6B",
        ),
        ModelVariant.QWEN_3_RERANKER_4B: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-Reranker-4B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_RERANKER_0_6B

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
            model="Qwen3Reranker",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _format_instruction(self, instruction, query, doc):
        """Format a query-document pair with instruction into the reranker input format."""
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            padding_side="left",
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Build the prompt with the reranker chat format
        prefix = (
            f"<|im_start|>system\n{self._SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        pairs = [
            self._format_instruction(self.sample_instruction, query, doc)
            for query, doc in self.sample_pairs
        ]

        max_length = 512

        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )

        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens

        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
