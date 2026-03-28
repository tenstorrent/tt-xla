# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Yi 1.5 model loader implementation for causal language modeling.
"""
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
    """Available Yi 1.5 model variants."""

    YI_1_5_34B_CHAT = "1.5_34B_Chat"
    YI_1_5_34B_CHAT_16K = "1.5_34B_Chat_16K"
    INFINITY_INSTRUCT_3M_0625_YI_1_5_9B = "Infinity_Instruct_3M_0625_Yi_1.5_9B"


class ModelLoader(ForgeModel):
    """Yi 1.5 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.YI_1_5_34B_CHAT: LLMModelConfig(
            pretrained_model_name="01-ai/Yi-1.5-34B-Chat",
            max_length=256,
        ),
        ModelVariant.YI_1_5_34B_CHAT_16K: LLMModelConfig(
            pretrained_model_name="01-ai/Yi-1.5-34B-Chat-16K",
            max_length=256,
        ),
        ModelVariant.INFINITY_INSTRUCT_3M_0625_YI_1_5_9B: LLMModelConfig(
            pretrained_model_name="BAAI/Infinity-Instruct-3M-0625-Yi-1.5-9B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.YI_1_5_34B_CHAT

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Yi1.5",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
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

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_tokens = self.tokenizer(
            self.sample_text,
            max_length=self._variant_config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return [input_tokens["input_ids"], input_tokens["attention_mask"]]
