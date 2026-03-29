# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama3 OpenBioLLM 70B AWQ model loader implementation for causal language modeling.

AWQ-quantized biomedical domain LLaMA 3 70B model from TitanML.
"""

import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Llama3 OpenBioLLM model variants."""

    LLAMA3_OPENBIOLLLM_70B_AWQ_4BIT = "Llama3_OpenBioLLM_70B_Awq_4bit"


class ModelLoader(ForgeModel):
    """Llama3 OpenBioLLM model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.LLAMA3_OPENBIOLLLM_70B_AWQ_4BIT: LLMModelConfig(
            pretrained_model_name="TitanML/Llama3-OpenBioLLM-70B-AWQ-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA3_OPENBIOLLLM_70B_AWQ_4BIT

    _AWQ_VARIANTS = frozenset({ModelVariant.LLAMA3_OPENBIOLLLM_70B_AWQ_4BIT})

    sample_text = "What are the common symptoms of type 2 diabetes?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Llama3-OpenBioLLM",
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
            **tokenizer_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self._variant in self._AWQ_VARIANTS:
            model_kwargs["device_map"] = "cpu"

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
