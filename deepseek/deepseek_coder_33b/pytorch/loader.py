# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Coder 33B model loader implementation for causal language modeling.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ....tools.utils import generate_no_cache, pad_inputs
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
    """Available DeepSeek Coder 33B model variants."""

    DEEPSEEK_33B_BASE = "33B_Base"
    DEEPSEEK_33B_INSTRUCT = "33B_Instruct"


class ModelLoader(ForgeModel):
    """DeepSeek Coder 33B model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_33B_BASE: LLMModelConfig(
            pretrained_model_name="deepseek-ai/deepseek-coder-33b-base",
            max_length=2048,
        ),
        ModelVariant.DEEPSEEK_33B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="deepseek-ai/deepseek-coder-33b-instruct",
            max_length=2048,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_33B_INSTRUCT

    sample_text = "write a quick sort algorithm in python."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek Coder 33B",
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
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if self._variant == ModelVariant.DEEPSEEK_33B_BASE:
            inputs = self.tokenizer(
                self.sample_text,
                return_tensors="pt",
            )["input_ids"]
        else:
            messages = [{"role": "user", "content": self.sample_text}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        padded_inputs, seq_len = pad_inputs(inputs)
        self.seq_len = seq_len

        return padded_inputs

    def decode_output(self, max_new_tokens, model, inputs, tokenizer):
        generated_text = generate_no_cache(
            max_new_tokens, model, inputs, self.seq_len, tokenizer
        )
        return generated_text
