# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 4 Scout NVFP4 model loader implementation for causal language modeling.

This loads the NVFP4-quantized variant of Llama 4 Scout 17B-16E Instruct
from RedHatAI, a Mixture of Experts model quantized to FP4 precision
using LLM Compressor with the compressed-tensors format.
"""

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Available Llama 4 Scout NVFP4 model variants for causal language modeling."""

    LLAMA4_SCOUT_17B_16E_INSTRUCT_NVFP4 = "Scout_17B_16E_Instruct_NVFP4"


class ModelLoader(ForgeModel):
    """Llama 4 Scout NVFP4 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.LLAMA4_SCOUT_17B_16E_INSTRUCT_NVFP4: LLMModelConfig(
            pretrained_model_name="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA4_SCOUT_17B_16E_INSTRUCT_NVFP4

    sample_text = "What are the key benefits of mixture of experts models?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Llama-4-Scout-17B-16E-Instruct-NVFP4",
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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = AutoConfig.from_pretrained(pretrained_model_name)

        # Reduce model dimensions for testing since the full 17B-16E
        # MoE model is too large to load directly.
        if self.num_layers is not None:
            config.text_config.num_hidden_layers = self.num_layers
        else:
            config.text_config.num_hidden_layers = 6
        config.text_config.num_attention_heads = 16
        config.text_config.hidden_size = 1024
        config.text_config.num_key_value_heads = 16
        config.text_config.intermediate_size = 1024 * 4
        config.text_config.num_local_experts = 16
        config.text_config.num_experts_per_tok = 1

        model_kwargs = {
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(config, **model_kwargs)

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
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
