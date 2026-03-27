# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mxbai-rerank-v2 model loader implementation for passage ranking.

This reranker uses a Qwen2 causal LM backbone with binary relevance scoring
(0/1 logits) to determine document relevance to a query.
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
    """Available mxbai-rerank-v2 model variants for passage ranking."""

    BASE_V2 = "base-v2"


class ModelLoader(ForgeModel):
    """mxbai-rerank-v2 model loader implementation for passage ranking.

    This reranker uses a Qwen2 causal LM backbone with a binary relevance
    scoring approach (0 for not relevant, 1 for relevant) to determine
    document relevance to a query.
    """

    _VARIANTS = {
        ModelVariant.BASE_V2: ModelConfig(
            pretrained_model_name="mixedbread-ai/mxbai-rerank-base-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_V2

    # Sample query-passage pairs for testing
    sample_pairs = [
        (
            "How many people live in Berlin?",
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        ),
    ]

    _RELEVANCE_PROMPT = (
        "You are a search relevance expert who evaluates how well documents "
        "match search queries. For each query-document pair, carefully analyze "
        "the semantic relationship between them, then provide your binary "
        "relevance judgment (0 for not relevant, 1 for relevant).\n"
        "Relevance:"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MxbaiRerankV2",
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

    def _format_input(self, query, document):
        """Format a query-document pair using the reranker's chat template."""
        user_content = (
            f"query: {query}\n" f"document: {document}\n" f"{self._RELEVANCE_PROMPT}"
        )
        messages = [
            {"role": "user", "content": user_content},
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
