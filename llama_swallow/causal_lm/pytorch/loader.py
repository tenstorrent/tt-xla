# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama-Swallow model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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


class ModelVariant(StrEnum):
    """Available Llama-Swallow model variants for causal language modeling."""

    LLAMA_3_1_SWALLOW_8B_V0_5 = "3.1_Swallow_8B_v0.5"
    LLAMA_3_1_SWALLOW_8B_INSTRUCT_V0_5 = "3.1_Swallow_8B_Instruct_v0.5"
    LLAMA_3_1_SWALLOW_70B_INSTRUCT_V0_3 = "3.1_Swallow_70B_Instruct_v0.3"


class ModelLoader(ForgeModel):
    """Llama-Swallow model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_1_SWALLOW_8B_V0_5: LLMModelConfig(
            pretrained_model_name="tokyotech-llm/Llama-3.1-Swallow-8B-v0.5",
            max_length=256,
        ),
        ModelVariant.LLAMA_3_1_SWALLOW_8B_INSTRUCT_V0_5: LLMModelConfig(
            pretrained_model_name="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
            max_length=256,
        ),
        ModelVariant.LLAMA_3_1_SWALLOW_70B_INSTRUCT_V0_3: LLMModelConfig(
            pretrained_model_name="tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_1_SWALLOW_70B_INSTRUCT_V0_3

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    sample_text = "Tokyo is the capital of"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.config = None
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Llama-Swallow",
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
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        self.load_config()

        model_kwargs = {
            "config": self.config,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if self._variant == ModelVariant.LLAMA_3_1_SWALLOW_8B_V0_5:
            inputs = self.tokenizer(
                self.sample_text,
                return_tensors="pt",
                padding="max_length",
                max_length=128,
            )
        else:
            inputs = self.tokenizer.apply_chat_template(
                self.messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length",
                max_length=128,
            )
        if (
            hasattr(self.model.config, "sliding_window")
            and self.model.config.sliding_window is not None
        ):
            self.model.config.sliding_window = inputs["input_ids"].shape[1]

        return inputs

    def get_mesh_config(self, num_devices: int):
        if self._variant in (
            ModelVariant.LLAMA_3_1_SWALLOW_8B_V0_5,
            ModelVariant.LLAMA_3_1_SWALLOW_8B_INSTRUCT_V0_5,
        ):
            mesh_shape = (1, num_devices)
        elif num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        elif num_devices == 8:  # llmbox
            mesh_shape = (2, 4)
        else:
            raise ValueError("Llama-Swallow 70B is only supported on llmbox and galaxy")

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        if self._variant in (
            ModelVariant.LLAMA_3_1_SWALLOW_8B_V0_5,
            ModelVariant.LLAMA_3_1_SWALLOW_8B_INSTRUCT_V0_5,
        ):
            return None

        shard_specs = {}

        shard_specs[model.model.embed_tokens.weight] = (None, "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        shard_specs[model.model.norm.weight] = ("batch",)

        for layer in model.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers

        return self.config
