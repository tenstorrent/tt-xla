# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mzwing NSFW 13B sft GGUF model loader implementation for causal language modeling.

Note: The Baichuan architecture is not supported for GGUF loading by transformers.
This loader uses the base Baichuan-13B model repo for config/tokenizer and loads
the model from config since GGUF weight loading is unsupported for this architecture.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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

# The base model repo for config and tokenizer (GGUF loading unsupported for baichuan)
BASE_MODEL_REPO = "baichuan-inc/Baichuan-13B-Base"


class ModelVariant(StrEnum):
    """Available NSFW 13B sft GGUF model variants for causal language modeling."""

    NSFW_13B_SFT_GGUF = "13B_SFT_GGUF"


class ModelLoader(ForgeModel):
    """mzwing NSFW 13B sft GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NSFW_13B_SFT_GGUF: LLMModelConfig(
            pretrained_model_name="mzwing/NSFW_13B_sft-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NSFW_13B_SFT_GGUF

    sample_text = "从前，"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="NSFW 13B sft GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_REPO, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self.load_config()
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

        inputs = self.tokenizer(
            prompts,
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
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        if self.config is not None:
            return self.config
        self.config = AutoConfig.from_pretrained(
            BASE_MODEL_REPO, trust_remote_code=True
        )
        return self.config
