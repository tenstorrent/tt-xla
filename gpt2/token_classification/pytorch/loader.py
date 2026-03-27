# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-2 LoRA model loader implementation for token classification.
"""

import torch
from transformers import AutoTokenizer, GPT2ForTokenClassification
from peft import PeftModel
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GPT-2 token classification model variants."""

    TINY_GPT2_TOKEN_CLS_LORA = (
        "peft-internal-testing/tiny_GPT2ForTokenClassification-lora"
    )


class ModelLoader(ForgeModel):
    """GPT-2 LoRA model loader for token classification tasks."""

    _VARIANTS = {
        ModelVariant.TINY_GPT2_TOKEN_CLS_LORA: ModelConfig(
            pretrained_model_name="peft-internal-testing/tiny_GPT2ForTokenClassification-lora",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_GPT2_TOKEN_CLS_LORA

    BASE_MODEL_NAME = "hf-internal-testing/tiny-random-gpt2"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GPT-2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        base_model = GPT2ForTokenClassification.from_pretrained(
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
            self._load_tokenizer()

        sample_text = "HuggingFace is a company based in Paris and New York"
        inputs = self.tokenizer(
            sample_text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )
        print(f"Predicted token class IDs: {predicted_token_class_ids}")
