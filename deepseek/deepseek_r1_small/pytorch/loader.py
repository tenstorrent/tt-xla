# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek R1 Small 2-layer model loader implementation for causal language modeling.

A pruned variant of DeepSeek-R1 reduced to 2 hidden layers (~2.4B parameters),
using the DeepSeek V3 MoE architecture with Multi-head Latent Attention.
"""

from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


class ModelLoader(ForgeModel):
    """DeepSeek R1 Small 2-layer model loader for causal language modeling."""

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = "silence09/DeepSeek-R1-Small-2layers"
        self.tokenizer = None
        self.text = "Please reason step by step. What is 25 multiplied by 16?"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-R1-Small-2layers",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
