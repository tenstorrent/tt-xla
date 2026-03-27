# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moonlight 16B-A3B-Instruct model loader implementation for causal language modeling.

Uses reduced MoE configuration for testing since the full 16B parameter
model is too large to load directly.
"""

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


class ModelLoader(ForgeModel):
    """Moonlight 16B-A3B-Instruct model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
        self.tokenizer = None
        self.text = "Give me a short introduction to large language model."
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Moonlight-16B-A3B-Instruct",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        # Reduce model dimensions for testing
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        else:
            config.num_hidden_layers = 6
        config.num_attention_heads = 16
        config.hidden_size = 1024
        config.num_key_value_heads = 16
        config.intermediate_size = 1024 * 4
        config.num_experts_per_tok = 2

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(config, **model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
