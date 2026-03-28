# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4 MoE Lite model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
from ....tools.utils import (
    cast_input_to_type,
    pad_inputs,
)


class ModelVariant(StrEnum):
    """Available GLM-4 MoE Lite model variants."""

    GLM_4_7_FLASH_REAP_23B_A3B = "4.7_Flash_REAP_23B_A3B"


class ModelLoader(ForgeModel):
    """GLM-4 MoE Lite model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_REAP_23B_A3B: LLMModelConfig(
            pretrained_model_name="cerebras/GLM-4.7-Flash-REAP-23B-A3B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH_REAP_23B_A3B

    sample_text = "Hey how are you doing today?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GLM_MoE_Lite",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        ).eval()

        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def decode_output(self, max_new_tokens, model, inputs, tokenizer):
        current_pos = self.seq_len

        for _ in range(max_new_tokens):
            logits = model(*inputs)

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            next_token_logits = logits[:, current_pos - 1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            inputs[0][:, current_pos] = next_token_id
            inputs[1][:, current_pos] = 1

            current_pos += 1

        valid_tokens = inputs[0][:, self.seq_len : current_pos].view(-1).tolist()
        answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)
        return answer

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )

        return self.config
