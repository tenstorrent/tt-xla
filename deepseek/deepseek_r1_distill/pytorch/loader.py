# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek R1 Distill model loader implementation for causal language modeling.

Supports distilled variants of DeepSeek-R1 that are compatible with
HuggingFace Transformers (the full 671B MoE model is not).
"""

from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DeepSeek R1 Distill model variants."""

    DISTILL_QWEN_1_5B = "Distill_Qwen_1_5B"
    DISTILL_QWEN_7B = "Distill_Qwen_7B"
    DISTILL_QWEN_7B_UNSLOTH_BNB_4BIT = "Distill_Qwen_7B_unsloth_bnb_4bit"
    DISTILL_QWEN_14B = "Distill_Qwen_14B"
    DISTILL_QWEN_14B_FP8_DYNAMIC = "Distill_Qwen_14B_FP8_dynamic"
    DISTILL_LLAMA_8B = "Distill_Llama_8B"
    DISTILL_LLAMA_70B = "Distill_Llama_70B"
    DISTILL_LLAMA_70B_BNB_4BIT = "Distill_Llama_70B_bnb_4bit"


class ModelLoader(ForgeModel):
    """DeepSeek R1 Distill model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.DISTILL_QWEN_1_5B: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_length=2048,
        ),
        ModelVariant.DISTILL_QWEN_7B: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            max_length=2048,
        ),
        ModelVariant.DISTILL_QWEN_7B_UNSLOTH_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
            max_length=2048,
        ),
        ModelVariant.DISTILL_QWEN_14B: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            max_length=2048,
        ),
        # RedHatAI FP8 dynamically quantized variant
        ModelVariant.DISTILL_QWEN_14B_FP8_DYNAMIC: LLMModelConfig(
            pretrained_model_name="RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic",
            max_length=2048,
        ),
        ModelVariant.DISTILL_LLAMA_8B: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            max_length=2048,
        ),
        ModelVariant.DISTILL_LLAMA_70B: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            max_length=2048,
        ),
        ModelVariant.DISTILL_LLAMA_70B_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit",
            max_length=2048,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILL_QWEN_1_5B

    sample_text = "Please reason step by step. What is 25 multiplied by 16?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    _AWQ_VARIANTS = frozenset({ModelVariant.DISTILL_LLAMA_70B_AWQ})

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek-R1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _is_gguf_variant(self):
        """Check if the current variant uses GGUF quantization."""
        return self._variant in self._GGUF_FILES

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        if self._is_gguf_variant():
            tokenizer_kwargs["gguf_file"] = self._gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        if self._variant in (ModelVariant.DISTILL_LLAMA_70B_BNB_4BIT,):
            model_kwargs["device_map"] = "cpu"
        model_kwargs |= kwargs

        # Quantized variants need device_map="cpu" for CPU-based loading
        if self._variant in (ModelVariant.DISTILL_QWEN_7B_UNSLOTH_BNB_4BIT,):
            model_kwargs["device_map"] = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
