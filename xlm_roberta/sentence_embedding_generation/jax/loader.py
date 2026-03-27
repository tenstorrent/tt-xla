# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multilingual E5 model loader implementation for sentence embedding generation.

Uses the XLM-RoBERTa architecture (intfloat/multilingual-e5-large) via
FlaxXLMRobertaModel for multilingual text embeddings.
"""

from typing import Optional

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
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available multilingual E5 model variants."""

    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"
    EMBEDIC_BASE = "djovak/embedic-base"


class ModelLoader(ForgeModel):
    """Multilingual E5 model loader for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.MULTILINGUAL_E5_LARGE: LLMModelConfig(
            pretrained_model_name="intfloat/multilingual-e5-large",
            max_length=128,
        ),
        ModelVariant.EMBEDIC_BASE: LLMModelConfig(
            pretrained_model_name="djovak/embedic-base",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTILINGUAL_E5_LARGE

    sample_text = "query: How is the weather today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="multilingual-e5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import FlaxXLMRobertaModel

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FlaxXLMRobertaModel.from_pretrained(
            self._model_name, from_pt=True, **model_kwargs
        )

        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        max_length = getattr(self._variant_config, "max_length", 128)

        inputs = self._tokenizer(
            self.sample_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="jax",
        )

        return inputs
