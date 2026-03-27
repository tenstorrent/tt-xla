# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama-Nemotron-Rerank model loader implementation for passage ranking.
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
    """Available Llama-Nemotron-Rerank model variants for passage ranking."""

    V2_1B = "v2_1B"


class ModelLoader(ForgeModel):
    """Llama-Nemotron-Rerank model loader implementation for passage ranking.

    This reranker uses a Llama 3.2 1B backbone with a binary classification head
    fine-tuned for ranking tasks. It uses bi-directional attention and a
    query-passage prompt template for scoring document relevance.
    """

    _VARIANTS = {
        ModelVariant.V2_1B: ModelConfig(
            pretrained_model_name="nvidia/llama-nemotron-rerank-1b-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_1B

    # Sample query-passage pairs for testing
    sample_pairs = [
        ("how much protein should a female eat", "hi"),
        (
            "how much protein should a female eat",
            "As a general guideline, the CDC's average requirement of protein "
            "for women ages 19 to 70 is 46 grams per day.",
        ),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Llama-Nemotron-Rerank",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _prompt_template(query, passage):
        """Format a query-passage pair using the reranker's prompt template."""
        return f"question:{query} \n \n passage:{passage}"

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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
            self._prompt_template(query, passage)
            for query, passage in self.sample_pairs
        ]

        inputs = self.tokenizer(
            texts,
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
