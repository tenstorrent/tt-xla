# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OPT LoRA model loader implementation for causal language modeling.

This model applies a LoRA adapter (peft-internal-testing/tiny-OPTForCausalLM-lora)
on top of the hf-internal-testing/tiny-random-OPTForCausalLM base model.
"""

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
    """Available OPT LoRA model variants."""

    TINY_OPT_LORA = "tiny-OPTForCausalLM-lora"


class ModelLoader(ForgeModel):
    """OPT LoRA model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.TINY_OPT_LORA: LLMModelConfig(
            pretrained_model_name="peft-internal-testing/tiny-OPTForCausalLM-lora",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_OPT_LORA

    BASE_MODEL_NAME = "hf-internal-testing/tiny-random-OPTForCausalLM"

    sample_text = "My name is Thomas and my main"

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
            model="OPT-LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL_NAME, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import OPTForCausalLM, AutoConfig
        from peft import PeftModel

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        base_model = OPTForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )
        model = PeftModel.from_pretrained(
            base_model, self._variant_config.pretrained_model_name, **kwargs
        )
        model = model.merge_and_unload()
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_tokens = self.tokenizer(
            self.sample_text,
            max_length=self._variant_config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return [input_tokens["input_ids"], input_tokens["attention_mask"]]
