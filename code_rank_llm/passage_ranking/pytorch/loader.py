# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CodeRankLLM model loader implementation for passage ranking.

CodeRankLLM is a 7B parameter LLM fine-tuned for listwise code reranking,
based on Qwen2.5-Coder-7B-Instruct. It scores multiple code passages
simultaneously to enhance code retrieval quality.
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
    """Available CodeRankLLM model variants for passage ranking."""

    CODE_RANK_LLM = "CodeRankLLM"


class ModelLoader(ForgeModel):
    """CodeRankLLM model loader implementation for passage ranking.

    This reranker uses a causal LM backbone fine-tuned for listwise code
    reranking, scoring multiple code passages for relevance to a query.
    """

    _VARIANTS = {
        ModelVariant.CODE_RANK_LLM: ModelConfig(
            pretrained_model_name="nomic-ai/CodeRankLLM",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CODE_RANK_LLM

    # Sample query-passage pairs for testing
    sample_pairs = [
        (
            "How to read a file in Python?",
            "def read_file(path):\n    with open(path, 'r') as f:\n        return f.read()",
        ),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CodeRankLLM",
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
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForCausalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        queries = [pair[0] for pair in self.sample_pairs]
        passages = [pair[1] for pair in self.sample_pairs]

        inputs = self.tokenizer(
            queries,
            passages,
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
