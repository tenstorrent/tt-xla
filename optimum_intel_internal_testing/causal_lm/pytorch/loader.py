# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Optimum Intel Internal Testing tiny-random-MptForCausalLM model loader
implementation for causal language modeling.
"""
from transformers import MptForCausalLM, AutoTokenizer
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
    """Available tiny-random-MptForCausalLM model variants."""

    TINY_RANDOM_MPT = "tiny_random_mpt"


class ModelLoader(ForgeModel):
    """Optimum Intel Internal Testing tiny-random-MptForCausalLM model loader for causal LM."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_MPT: LLMModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-MptForCausalLM",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_MPT

    sample_text = "My name is Thomas and my main"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Optimum_Intel_Internal_Testing",
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
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MptForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)
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
