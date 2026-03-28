# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PyLingual model loader implementation for text2text code generation.
"""

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available PyLingual model variants."""

    PY39_V1_STATEMENT = "py39_v1_statement"


class ModelLoader(ForgeModel):
    """PyLingual model loader implementation for text2text code generation tasks."""

    _VARIANTS = {
        ModelVariant.PY39_V1_STATEMENT: LLMModelConfig(
            pretrained_model_name="syssec-utd/py39-pylingual-v1-statement",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PY39_V1_STATEMENT

    sample_text = "def hello_world():"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PyLingual",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_tensor = torch.tensor(
            self._cached_model.generation_config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
