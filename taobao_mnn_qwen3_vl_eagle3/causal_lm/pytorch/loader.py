# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Taobao-MNN Qwen3-VL-8B-Instruct-Eagle3 model loader implementation for causal language modeling.

This is an EAGLE-3 speculative decoding draft model based on Llama architecture,
designed to accelerate inference of Qwen/Qwen3-VL-8B-Instruct.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
    """Available Qwen3-VL Eagle3 model variants for causal LM."""

    QWEN3_VL_8B_INSTRUCT_EAGLE3 = "8b_instruct_eagle3"


class ModelLoader(ForgeModel):
    """Taobao-MNN Qwen3-VL-8B-Instruct-Eagle3 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_8B_INSTRUCT_EAGLE3: LLMModelConfig(
            pretrained_model_name="taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_8B_INSTRUCT_EAGLE3

    sample_text = "What is your favorite city?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="taobao_mnn_qwen3_vl_eagle3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding=True,
            truncation=True,
        )
        return inputs
