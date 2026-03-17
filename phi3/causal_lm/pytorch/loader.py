# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi-3 causal language modeling loader
"""
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    MINI_128K = "Mini_128K_Instruct"
    MINI_4K = "Mini_4K_Instruct"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.MINI_128K: ModelConfig(
            pretrained_model_name="microsoft/Phi-3-mini-128k-instruct"
        ),
        ModelVariant.MINI_4K: ModelConfig(
            pretrained_model_name="microsoft/Phi-3-mini-4k-instruct"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINI_128K

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Phi-3",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def load_model(self, *, dtype_override=None, **kwargs):
        self._ensure_tokenizer()

        model_kwargs = {"use_cache": False}
        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name,
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, prompt: Optional[str] = None):
        self._ensure_tokenizer()
        input_prompt = [
            {
                "role": "user",
                "content": prompt
                or "Can you provide ways to eat combinations of bananas and dragonfruits?",
            }
        ]
        text = self.tokenizer.apply_chat_template(
            input_prompt, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)

        return [input_ids, attn_mask]
