# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OPT LoRA model loader implementation for question answering.
"""
import torch
from transformers import OPTForQuestionAnswering, AutoTokenizer
from peft import PeftModel
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available OPT LoRA QA model variants."""

    TINY_OPT_QA_LORA = "tiny_OPTForQuestionAnswering_lora"


class ModelLoader(ForgeModel):
    """OPT LoRA model loader for question answering tasks."""

    _VARIANTS = {
        ModelVariant.TINY_OPT_QA_LORA: ModelConfig(
            pretrained_model_name="peft-internal-testing/tiny_OPTForQuestionAnswering-lora",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_OPT_QA_LORA

    BASE_MODEL_NAME = "hf-internal-testing/tiny-random-OPTForCausalLM"

    sample_question = "Who was Jim Henson?"
    sample_context = "Jim Henson was a nice puppet"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="OPT-QA-LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL_NAME, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        base_model = OPTForQuestionAnswering.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_tokens = self.tokenizer(
            self.sample_question,
            self.sample_context,
            max_length=32,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return [input_tokens["input_ids"], input_tokens["attention_mask"]]
