# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DistilRoBERTa model loader implementation for masked language modeling.
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
    """Available DistilRoBERTa model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """DistilRoBERTa model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="distilbert/distilroberta-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_text = "The capital of France is <mask>."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DistilRoBERTa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
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
        from transformers import FlaxRobertaForMaskedLM

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FlaxRobertaForMaskedLM.from_pretrained(self._model_name, **model_kwargs)

        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return inputs
