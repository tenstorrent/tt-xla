# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE Reranker v2 Gemma model loader implementation for passage ranking.
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
    """Available BGE Reranker v2 Gemma model variants for passage ranking."""

    GEMMA = "gemma"


class ModelLoader(ForgeModel):
    """BGE Reranker v2 Gemma model loader implementation for passage ranking.

    This reranker uses a causal LM backbone (Gemma-2B) with a yes/no logit
    scoring approach to determine document relevance to a query.
    """

    _VARIANTS = {
        ModelVariant.GEMMA: ModelConfig(
            pretrained_model_name="BAAI/bge-reranker-v2-gemma",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA

    # Sample query-passage pairs for testing
    sample_pairs = [
        ("what is panda?", "hi"),
        (
            "what is panda?",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        ),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BGE-Reranker-v2-Gemma",
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
            return_tensors="pt",
            max_length=512,
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
