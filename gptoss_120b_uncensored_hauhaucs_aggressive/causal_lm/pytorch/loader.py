# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPTOSS-120B-Uncensored-HauhauCS-Aggressive GGUF model loader implementation for causal language modeling.
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


class ModelVariant(StrEnum):
    """Available GPTOSS-120B-Uncensored-HauhauCS-Aggressive model variants."""

    GPTOSS_120B_UNCENSORED_HAUHAUCS_AGGRESSIVE = "120B_Uncensored_HauhauCS_Aggressive"


class ModelLoader(ForgeModel):
    """GPTOSS-120B-Uncensored-HauhauCS-Aggressive GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GPTOSS_120B_UNCENSORED_HAUHAUCS_AGGRESSIVE: LLMModelConfig(
            pretrained_model_name="HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPTOSS_120B_UNCENSORED_HAUHAUCS_AGGRESSIVE

    _GGUF_FILES = {
        ModelVariant.GPTOSS_120B_UNCENSORED_HAUHAUCS_AGGRESSIVE: "GPTOSS-120B-Uncensored-HauhauCS-Aggressive-MXFP4.gguf",
    }

    @property
    def GGUF_FILE(self):
        return self._GGUF_FILES[self._variant]

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

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
            model="GPTOSS-120B-Uncensored-HauhauCS-Aggressive",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "gguf_file": self.GGUF_FILE,
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name,
                gguf_file=self.GGUF_FILE,
                trust_remote_code=True,
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.config = model.config
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        elif num_devices == 8:  # llmbox
            mesh_shape = (2, 4)
        else:
            raise ValueError(
                "GPTOSS-120B-Uncensored-HauhauCS-Aggressive is only supported on llmbox and galaxy"
            )

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}

        shard_specs[model.model.embed_tokens.weight] = (None, "batch")
        shard_specs[model.model.norm.weight] = ("batch",)
        shard_specs[model.lm_head.weight] = ("model", "batch")

        for layer in model.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.o_proj.bias] = ("batch",)
            shard_specs[layer.self_attn.sinks] = (None,)

            shard_specs[layer.mlp.router.weight] = (None, "batch")
            shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch", None)
            shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
            shard_specs[layer.mlp.experts.down_proj] = ("model", None, "batch")
            shard_specs[layer.mlp.experts.down_proj_bias] = ("model", "batch")

            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
            trust_remote_code=True,
        )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers

        return self.config
