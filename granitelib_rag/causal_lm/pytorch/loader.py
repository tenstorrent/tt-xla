# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GraniteLib RAG model loader implementation for causal language modeling.

ibm-granite/granitelib-rag-r1.0 is a collection of LoRA adapters built on
ibm-granite/granite-4.0-micro for RAG pipeline tasks including query rewrite,
context relevance, answerability determination, hallucination detection,
citation generation, and query clarification.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional
from dataclasses import dataclass

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


@dataclass
class GraniteLibRAGConfig(LLMModelConfig):
    """Configuration for GraniteLib RAG model variants."""

    base_model_name: str = "ibm-granite/granite-4.0-micro"
    adapter_subfolder: Optional[str] = None


class ModelVariant(StrEnum):
    """Available GraniteLib RAG model variants."""

    ANSWERABILITY_DETERMINATION = "answerability-determination"
    CONTEXT_RELEVANCE = "context-relevance"
    HALLUCINATION_DETECTION = "hallucination-detection"
    QUERY_REWRITE = "query-rewrite"
    CITATION_GENERATION = "citation-generation"
    QUERY_CLARIFICATION = "query-clarification"


class ModelLoader(ForgeModel):
    """GraniteLib RAG model loader implementation for causal language modeling."""

    _VARIANTS = {
        ModelVariant.ANSWERABILITY_DETERMINATION: GraniteLibRAGConfig(
            pretrained_model_name="ibm-granite/granitelib-rag-r1.0",
            base_model_name="ibm-granite/granite-4.0-micro",
            adapter_subfolder="answerability_determination",
            max_length=128,
        ),
        ModelVariant.CONTEXT_RELEVANCE: GraniteLibRAGConfig(
            pretrained_model_name="ibm-granite/granitelib-rag-r1.0",
            base_model_name="ibm-granite/granite-4.0-micro",
            adapter_subfolder="context_relevance",
            max_length=128,
        ),
        ModelVariant.HALLUCINATION_DETECTION: GraniteLibRAGConfig(
            pretrained_model_name="ibm-granite/granitelib-rag-r1.0",
            base_model_name="ibm-granite/granite-4.0-micro",
            adapter_subfolder="hallucination_detection",
            max_length=128,
        ),
        ModelVariant.QUERY_REWRITE: GraniteLibRAGConfig(
            pretrained_model_name="ibm-granite/granitelib-rag-r1.0",
            base_model_name="ibm-granite/granite-4.0-micro",
            adapter_subfolder="query_rewrite",
            max_length=128,
        ),
        ModelVariant.CITATION_GENERATION: GraniteLibRAGConfig(
            pretrained_model_name="ibm-granite/granitelib-rag-r1.0",
            base_model_name="ibm-granite/granite-4.0-micro",
            adapter_subfolder="citation_generation",
            max_length=128,
        ),
        ModelVariant.QUERY_CLARIFICATION: GraniteLibRAGConfig(
            pretrained_model_name="ibm-granite/granitelib-rag-r1.0",
            base_model_name="ibm-granite/granite-4.0-micro",
            adapter_subfolder="query_clarification",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANSWERABILITY_DETERMINATION

    sample_text = "What is retrieval augmented generation?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GraniteLib-RAG",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        cfg = self._variant_config
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        cfg = self._variant_config

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(cfg.base_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        cfg = self._variant_config
        self.config = AutoConfig.from_pretrained(cfg.base_model_name)

        return self.config
