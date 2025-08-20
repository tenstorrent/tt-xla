# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Math model loader implementation for causal language modeling.
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
    """Available DeepSeek Math model variants."""

    DEEPSEEK_7B_INSTRUCT = "7b_instruct"


class ModelLoader(ForgeModel):
    """DeepSeek Math model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_7B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="deepseek-ai/deepseek-math-7b-instruct",
            max_length=2048,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_7B_INSTRUCT

    sample_text = (
        "what is the integral of x^2 from 0 to 2?\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with a specified variant."""
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Return model info for the selected variant."""
        return ModelInfo(
            model="deepseek_math",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_QA,
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

    def load_model(self, dtype_override=None):
        """Load and return the DeepSeek Math model."""
        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.generation_config = GenerationConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        model.generation_config.use_cache = False  # Disable KV cache

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare and return tokenized inputs using the sample prompt."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
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
