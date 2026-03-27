# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CTXL-Rerank model loader implementation for passage ranking.
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
    """Available CTXL-Rerank model variants for passage ranking."""

    CTXL_RERANK_V2_INSTRUCT_MULTILINGUAL_1B = "v2_instruct_multilingual_1B"


class ModelLoader(ForgeModel):
    """CTXL-Rerank model loader implementation for passage ranking.

    This reranker uses a causal LM backbone (Qwen3) with a prompt-based scoring
    approach. Relevance scores are extracted from the logit at vocab index 0
    of the last token position.
    """

    _VARIANTS = {
        ModelVariant.CTXL_RERANK_V2_INSTRUCT_MULTILINGUAL_1B: ModelConfig(
            pretrained_model_name="ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CTXL_RERANK_V2_INSTRUCT_MULTILINGUAL_1B

    # Sample query-passage pairs for testing
    sample_pairs = [
        (
            "How many people live in Berlin?",
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        ),
    ]

    # Prompt template per the model card
    _PROMPT_TEMPLATE = (
        "Check whether a given document contains information helpful to answer the query.\n"
        "<Document> {document}\n"
        "<Query> {query} {instruction} ??"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CTXL-Rerank",
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
        )
        return self.tokenizer

    def _format_input(self, query, document, instruction=""):
        """Format a query-document pair using the reranker's prompt template."""
        return self._PROMPT_TEMPLATE.format(
            document=document,
            query=query,
            instruction=instruction,
        )

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
