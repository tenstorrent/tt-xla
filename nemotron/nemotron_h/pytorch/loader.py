# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron-H model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Available Nemotron-H model variants for causal language modeling."""

    NEMOTRON_H_4B_INSTRUCT_128K = "H_4B_Instruct_128K"
    NEMOTRON_NANO_9B_V2_FP8_DYNAMIC = "Nano_9B_v2_FP8_dynamic"
    NEMOTRON_NANO_12B_V2 = "Nano_12B_v2"
    NEMOTRON_NANO_12B_V2_BASE = "Nano_12B_v2_Base"


class ModelLoader(ForgeModel):
    """Nemotron-H model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_H_4B_INSTRUCT_128K: LLMModelConfig(
            pretrained_model_name="nvidia/Nemotron-H-4B-Instruct-128K",
            max_length=128,
        ),
        ModelVariant.NEMOTRON_NANO_9B_V2_FP8_DYNAMIC: LLMModelConfig(
            pretrained_model_name="RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic",
            max_length=128,
        ),
        ModelVariant.NEMOTRON_NANO_12B_V2: LLMModelConfig(
            pretrained_model_name="nvidia/NVIDIA-Nemotron-Nano-12B-v2",
            max_length=128,
        ),
        ModelVariant.NEMOTRON_NANO_12B_V2_BASE: LLMModelConfig(
            pretrained_model_name="nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_H_4B_INSTRUCT_128K

    sample_text = "Give me a short introduction to large language model."

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
            model="Nemotron-H",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True}
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

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        # Base models use plain text; chat models use chat template
        if self._variant in (ModelVariant.NEMOTRON_NANO_12B_V2_BASE,):
            prompts = [self.sample_text]
        else:
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
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
