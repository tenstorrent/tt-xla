# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 1.5 MoE model loader implementation for causal language modeling.
"""
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

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
    """Available Qwen 1.5 MoE model variants for causal language modeling."""

    QWEN_1_5_MOE_A2_7B_CHAT_W4A16 = "A2.7B-Chat-quantized-w4a16"


class ModelLoader(ForgeModel):
    """Qwen 1.5 MoE model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_1_5_MOE_A2_7B_CHAT_W4A16: LLMModelConfig(
            pretrained_model_name="nm-testing/Qwen1.5-MoE-A2.7B-Chat-quantized.w4a16",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_1_5_MOE_A2_7B_CHAT_W4A16

    chat_messages = [
        {"role": "system", "content": "You are Jim Keller, the CEO of Tenstorrent"},
        {"role": "user", "content": "Introduce yourself please!"},
    ]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 1.5 MoE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def load_model(self, *, dtype_override=None, **kwargs):
        self._ensure_tokenizer()

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "cpu",
        }

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model._supports_cache_class = False
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        self._ensure_tokenizer()

        max_length = self._variant_config.max_length

        batch_messages = [self.chat_messages] * batch_size
        prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch_messages
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return inputs
