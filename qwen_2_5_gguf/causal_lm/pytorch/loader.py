# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 GGUF model loader implementation for causal language modeling.
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
    """Available Qwen 2.5 GGUF model variants for causal language modeling."""

    QWEN_2_5_1_5B_INSTRUCT_GGUF = "1.5B_Instruct_GGUF"
    BARTOWSKI_QWEN_2_5_1_5B_INSTRUCT_GGUF = "Bartowski_1.5B_Instruct_GGUF"
    LMSTUDIO_QWEN_2_5_7B_INSTRUCT_1M_GGUF = "Lmstudio_7B_Instruct_1M_GGUF"
    QUANTFACTORY_QWEN_2_5_7B_INSTRUCT_ABLITERATED_V2_GGUF = (
        "QuantFactory_7B_Instruct_abliterated_v2_GGUF"
    )


class ModelLoader(ForgeModel):
    """Qwen 2.5 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_5_7B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_3B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen2.5-3B-Instruct-GGUF",
            max_length=128,
        ),
        ModelVariant.BARTOWSKI_QWEN_2_5_1_5B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
            max_length=128,
        ),
        ModelVariant.LMSTUDIO_QWEN_2_5_7B_INSTRUCT_1M_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF",
            max_length=128,
        ),
        ModelVariant.QUANTFACTORY_QWEN_2_5_7B_INSTRUCT_ABLITERATED_V2_GGUF: LLMModelConfig(
            pretrained_model_name="QuantFactory/Qwen2.5-7B-Instruct-abliterated-v2-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_7B_INSTRUCT_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_2_5_1_5B_INSTRUCT_GGUF: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        ModelVariant.BARTOWSKI_QWEN_2_5_1_5B_INSTRUCT_GGUF: "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        ModelVariant.LMSTUDIO_QWEN_2_5_7B_INSTRUCT_1M_GGUF: "Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf",
        ModelVariant.QUANTFACTORY_QWEN_2_5_7B_INSTRUCT_ABLITERATED_V2_GGUF: "Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_M.gguf",
    }

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers
        self.gguf_file = self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 2.5 GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
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
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

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
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
