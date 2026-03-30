# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TruthfulQA informativeness judge model loader implementation for causal language modeling.

Loads the allenai/truthfulqa-info-judge-llama2-7B model, a LLaMA 2 7B fine-tuned
to judge whether an answer to a TruthfulQA question is informative (outputs "yes" or "no").

Available variants:
- TRUTHFULQA_INFO_JUDGE_LLAMA2_7B: allenai/truthfulqa-info-judge-llama2-7B
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    """Available TruthfulQA info judge model variants for causal language modeling."""

    TRUTHFULQA_INFO_JUDGE_LLAMA2_7B = "truthfulqa_info_judge_llama2_7b"


class ModelLoader(ForgeModel):
    """TruthfulQA informativeness judge model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TRUTHFULQA_INFO_JUDGE_LLAMA2_7B: LLMModelConfig(
            pretrained_model_name="allenai/truthfulqa-info-judge-llama2-7B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TRUTHFULQA_INFO_JUDGE_LLAMA2_7B

    sample_text = "Q: What is the capital of France?\nA: The capital of France is Paris.\nHelpful:"

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
            model="TruthfulQAInfoJudgeLlama2_7B",
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
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            return_token_type_ids=False,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
