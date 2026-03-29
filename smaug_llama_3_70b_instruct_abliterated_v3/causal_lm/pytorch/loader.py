# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Smaug Llama 3 70B Instruct Abliterated v3 model loader implementation
"""
import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Smaug Llama 3 70B Instruct Abliterated v3 model variants."""

    SMAUG_LLAMA_3_70B_INSTRUCT_ABLITERATED_V3 = (
        "Smaug-Llama-3-70B-Instruct-abliterated-v3"
    )


class ModelLoader(ForgeModel):
    """Smaug Llama 3 70B Instruct Abliterated v3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.SMAUG_LLAMA_3_70B_INSTRUCT_ABLITERATED_V3: LLMModelConfig(
            pretrained_model_name="failspy/Smaug-Llama-3-70B-Instruct-abliterated-v3",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMAUG_LLAMA_3_70B_INSTRUCT_ABLITERATED_V3

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
            model="Smaug Llama 3 70B Instruct Abliterated v3",
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

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length
        prompt = "Hey how are you doing today?"

        tokenized_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )

        inputs = {
            "input_ids": tokenized_inputs.input_ids,
            "attention_mask": tokenized_inputs.attention_mask,
        }

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        token_ids = torch.argmax(logits, dim=-1)
        decoded = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        return decoded[0] if len(decoded) == 1 else decoded
