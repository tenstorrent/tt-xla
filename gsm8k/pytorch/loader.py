# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GSM8K LoRA model loader implementation for causal language modeling.

This model is a LoRA adapter fine-tuned on the GSM8K math dataset,
applied on top of Mistral-7B-v0.1.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Available GSM8K LoRA model variants."""

    GSM8K = "gsm8k"


class ModelLoader(ForgeModel):
    """GSM8K LoRA model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GSM8K: ModelConfig(
            pretrained_model_name="predibase/gsm8k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GSM8K

    BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"

    sample_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GSM8K-LoRA",
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

        base_model = AutoModelForCausalLM.from_pretrained(
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
            self.sample_text,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return [input_tokens["input_ids"], input_tokens["attention_mask"]]

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
