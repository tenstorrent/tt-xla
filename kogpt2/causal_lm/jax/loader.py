# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
KoGPT2 model loader implementation for causal language modeling.
"""

from typing import Optional

import jax.numpy as jnp

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


class ModelVariant(StrEnum):
    """Available KoGPT2 model variants."""

    BASE_V2 = "Base V2"


class ModelLoader(ForgeModel):
    """KoGPT2 model loader implementation for causal language modeling."""

    _VARIANTS = {
        ModelVariant.BASE_V2: LLMModelConfig(
            pretrained_model_name="skt/kogpt2-base-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_V2

    sample_text = "안녕하세요, 오늘 날씨가"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="KoGPT2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import FlaxGPT2LMHeadModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FlaxGPT2LMHeadModel.from_pretrained(
            self._model_name, from_pt=True, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer()

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return {"input_ids": inputs.input_ids}
