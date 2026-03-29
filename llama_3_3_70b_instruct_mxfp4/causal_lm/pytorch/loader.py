# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AMD Llama 3.3 70B Instruct MXFP4 model loader implementation for causal language modeling.

Supports the AMD MXFP4-quantized variant of Meta Llama 3.3 70B Instruct,
quantized using AMD-Quark with OCP Microscaling FP4 format.
"""

from typing import Optional

import torch
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
    """Available AMD Llama 3.3 70B Instruct MXFP4 model variants for causal language modeling."""

    LLAMA_3_3_70B_INSTRUCT_MXFP4 = "70B_Instruct_MXFP4"


class ModelLoader(ForgeModel):
    """AMD Llama 3.3 70B Instruct MXFP4 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_3_70B_INSTRUCT_MXFP4: LLMModelConfig(
            pretrained_model_name="amd/Llama-3.3-70B-Instruct-MXFP4-Preview",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_3_70B_INSTRUCT_MXFP4

    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Llama 3.3 70B Instruct MXFP4",
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
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        ).eval()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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

    def get_mesh_config(self, num_devices: int):
        if num_devices == 32:
            mesh_shape = (4, 8)
        else:
            mesh_shape = (2, num_devices // 2)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        shard_specs[model.model.embed_tokens.weight] = (None, "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        shard_specs[model.model.norm.weight] = ("batch",)
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)
        return shard_specs
