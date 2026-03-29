# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
nm-testing/asym-w8w8-int8-static-per-tensor-tiny-llama causal language modeling loader
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
    """Available asym-w8w8-int8-static-per-tensor-tiny-llama model variants."""

    ASYM_W8W8_INT8_STATIC_PER_TENSOR_TINY_LLAMA = (
        "asym_w8w8_int8_static_per_tensor_tiny_llama"
    )


class ModelLoader(ForgeModel):
    """asym-w8w8-int8-static-per-tensor-tiny-llama model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.ASYM_W8W8_INT8_STATIC_PER_TENSOR_TINY_LLAMA: LLMModelConfig(
            pretrained_model_name="nm-testing/asym-w8w8-int8-static-per-tensor-tiny-llama",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ASYM_W8W8_INT8_STATIC_PER_TENSOR_TINY_LLAMA

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model metadata."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="asym-w8w8-int8-static-per-tensor-tiny-llama",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        ).eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
