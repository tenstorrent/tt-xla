# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 Next model loader implementation for causal language modeling.
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
    """Available Qwen 3 Next model variants for causal language modeling."""

    QWEN_3_NEXT_80B_A3B_INSTRUCT = "80B_A3B_Instruct"
    QWEN_3_NEXT_80B_A3B_INSTRUCT_MLX_6BIT = "80B_A3B_Instruct_MLX_6bit"


class ModelLoader(ForgeModel):
    """Qwen 3 Next model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_NEXT_80B_A3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-Next-80B-A3B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_3_NEXT_80B_A3B_INSTRUCT_MLX_6BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-6bit",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_NEXT_80B_A3B_INSTRUCT

    # Variants with NVFP4 quantized weights require ignore_mismatched_sizes
    # because the packed FP4 weight shapes differ from the model definition.
    _NVFP4_VARIANTS = {ModelVariant.QWEN_3_NEXT_80B_A3B_INSTRUCT_NVFP4}

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language model."

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
            model="Qwen 3 Next",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _is_gguf_variant(self):
        """Check if the current variant uses GGUF quantization."""
        return self._variant in self._GGUF_FILES

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        if self._is_gguf_variant():
            tokenizer_kwargs["gguf_file"] = self._gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        if self._variant in self._NVFP4_VARIANTS:
            model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs |= kwargs

        if self._is_gguf_variant():
            model_kwargs["gguf_file"] = self._gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
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

        messages = [{"role": "user", "content": self.sample_text}]
        chat_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self._variant == ModelVariant.QWEN_3_NEXT_80B_A3B_THINKING:
            chat_kwargs["enable_thinking"] = True
        text = self.tokenizer.apply_chat_template(
            messages,
            **chat_kwargs,
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

    def load_config(self):
        config_kwargs = {}
        if self._is_gguf_variant():
            config_kwargs["gguf_file"] = self._gguf_file

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, **config_kwargs
        )

        return self.config
