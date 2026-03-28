# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M2.5-REAP NVFP4 model loader implementation for causal language modeling.

This loads the REAP-pruned (154 experts from 256) and NVFP4-quantized variant
of MiniMax-M2.5, using the native transformers MiniMaxForCausalLM class.
"""
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.models.minimax.configuration_minimax import MiniMaxConfig
from transformers.models.minimax.modeling_minimax import MiniMaxForCausalLM
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
    """Available MiniMax-M2.5-REAP NVFP4 model variants for causal language modeling."""

    MINIMAX_M2_5_REAP_NVFP4 = "M2.5_REAP_NVFP4"


class ModelLoader(ForgeModel):
    """MiniMax-M2.5-REAP NVFP4 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MINIMAX_M2_5_REAP_NVFP4: LLMModelConfig(
            pretrained_model_name="lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINIMAX_M2_5_REAP_NVFP4

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
            model="MiniMax-M2.5-REAP NVFP4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_native_config(self, base_config=None):
        """Build a native MiniMaxConfig from the remote config.

        The REAP model uses model_type 'minimax_m2' with trust_remote_code,
        but the native transformers 'minimax' architecture is compatible.
        This converts the config to avoid trust_remote_code for model loading.
        """
        if base_config is None:
            base_config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, trust_remote_code=True
            )

        config_dict = base_config.to_dict()
        config_dict.pop("auto_map", None)
        config_dict.pop("model_type", None)
        config_dict.pop("transformers_version", None)

        if self.num_layers is not None:
            config_dict["num_hidden_layers"] = self.num_layers

        return MiniMaxConfig(**config_dict)

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
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
        # NVFP4 packed weight shapes differ from the model definition
        model_kwargs["ignore_mismatched_sizes"] = True
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

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
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

    def load_config(self):
        self.config = self._build_native_config()
        return self.config
