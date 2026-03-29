# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenBuddy model loader implementation for causal language modeling.
"""
import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available OpenBuddy model variants for causal language modeling."""

    OPENBUDDY_MISTRAL_22B_V21_1_32K = "OpenBuddy_Mistral_22B_v21.1_32K"
    OPENBUDDY_ZERO_56B_V21_2_32K = "OpenBuddy_Zero_56B_v21.2_32K"


class ModelLoader(ForgeModel):
    """OpenBuddy model loader implementation."""

    _VARIANTS = {
        ModelVariant.OPENBUDDY_MISTRAL_22B_V21_1_32K: LLMModelConfig(
            pretrained_model_name="OpenBuddy/openbuddy-mistral-22b-v21.1-32k",
            max_length=256,
        ),
        ModelVariant.OPENBUDDY_ZERO_56B_V21_2_32K: LLMModelConfig(
            pretrained_model_name="OpenBuddy/openbuddy-zero-56b-v21.2-32k",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENBUDDY_ZERO_56B_V21_2_32K

    sample_text = "What is the capital of France?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="openbuddy",
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
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
