# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SearchR1 Qwen2.5-7B EM-PPO model loader implementation for causal language modeling.

SearchR1 is a search-augmented reasoning model trained with PPO on NQ and HotpotQA datasets,
based on Qwen2.5-7B.
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


class ModelLoader(ForgeModel):
    """SearchR1 Qwen2.5-7B EM-PPO model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3"
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="SearchR1-Qwen2.5-7B-EM-PPO",
            variant=variant_name,
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
            self.model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = Qwen2ForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        model.eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [
            {"role": "user", "content": "What is the capital of France?"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
