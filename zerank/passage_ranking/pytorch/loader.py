# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ZeRank model loader implementation for passage ranking.
"""
import torch
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
    """Available ZeRank model variants for passage ranking."""

    ZERANK_2 = "zerank_2"


class ModelLoader(ForgeModel):
    """ZeRank model loader implementation for passage ranking.

    This reranker uses a Qwen3 causal LM backbone with a yes/no logit scoring
    approach to determine document relevance to a query.
    """

    _VARIANTS = {
        ModelVariant.ZERANK_2: ModelConfig(
            pretrained_model_name="zeroentropy/zerank-2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ZERANK_2

    # Sample query-passage pairs for testing
    sample_pairs = [
        (
            "What is 2+2?",
            "4",
        ),
        (
            "What is 2+2?",
            "The answer is definitely 1 million",
        ),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ZeRank",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        return self.tokenizer

    def _format_input(self, query, document):
        """Format a query-document pair using the reranker's chat template."""
        messages = [
            {"role": "system", "content": f"Query: {query}"},
            {"role": "user", "content": document},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForCausalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        texts = [
            self._format_input(query, document) for query, document in self.sample_pairs
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
