# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek LLM model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Optional
from ....tools.utils import generate_no_cache, pad_inputs
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
    """Available DeepSeek LLM model variants."""

    DEEPSEEK_67B_BASE = "67B_Base"


class ModelLoader(ForgeModel):
    """DeepSeek LLM model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_67B_BASE: LLMModelConfig(
            pretrained_model_name="deepseek-ai/deepseek-llm-67b-base",
            max_length=4096,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_67B_BASE

    sample_text = "The future of artificial intelligence is"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with a specified variant."""
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Return model info for the selected variant."""
        return ModelInfo(
            model="DeepSeek",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load and return tokenizer for the model."""
        tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

        return tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepSeek LLM model."""
        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **kwargs,
        )
        model.generation_config = GenerationConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            **kwargs,
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        model.generation_config.use_cache = False

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare and return tokenized inputs using the sample prompt."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_ids = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        ).input_ids
        inputs, seq_len = pad_inputs(input_ids)
        self.seq_len = seq_len
        return inputs

    def decode_output(self, max_new_tokens, model, inputs, tokenizer):
        """Generate text output using no-cache generation loop."""
        return generate_no_cache(
            max_new_tokens=max_new_tokens,
            model=model,
            input_ids=inputs,
            seq_len=self.seq_len,
            tokenizer=tokenizer,
        )
