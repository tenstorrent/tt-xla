# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M2.5 MLX model loader implementation for causal language modeling.

Uses the native transformers MiniMaxForCausalLM class rather than
trust_remote_code, since the remote modeling code requires a newer
transformers version than is currently available.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.minimax.configuration_minimax import MiniMaxConfig
from transformers.models.minimax.modeling_minimax import MiniMaxForCausalLM

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
    """Available MiniMax-M2.5 MLX model variants."""

    MINIMAX_M2_5_MLX_6BIT = "M2.5_MLX_6bit"


class ModelLoader(ForgeModel):
    """MiniMax-M2.5 MLX model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MINIMAX_M2_5_MLX_6BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/MiniMax-M2.5-MLX-6bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINIMAX_M2_5_MLX_6BIT

    messages = [
        {
            "role": "user",
            "content": "Give me a short introduction to large language model.",
        },
    ]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MiniMax-M2.5 MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_native_config(self, base_config=None):
        """Build a native MiniMaxConfig from the remote minimax_m2 config.

        The M2.5 model uses model_type 'minimax_m2' with trust_remote_code,
        but the native transformers 'minimax' architecture is compatible.
        This converts the config to avoid trust_remote_code for model loading.
        """
        if base_config is None:
            base_config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, trust_remote_code=True
            )

        config_dict = base_config.to_dict()
        # Remove remote-code specific fields
        config_dict.pop("auto_map", None)
        config_dict.pop("model_type", None)
        config_dict.pop("transformers_version", None)

        if self.num_layers is not None:
            config_dict["num_hidden_layers"] = self.num_layers

        return MiniMaxConfig(**config_dict)

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **tokenizer_kwargs
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

        config = self._build_native_config()
        model_kwargs["config"] = config

        model = MiniMaxForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=self._variant_config.max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        self.config = self._build_native_config()
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers
        return self.config
