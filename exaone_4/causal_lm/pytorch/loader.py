# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EXAONE 4.0 model loader implementation for causal language modeling.
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
    """Available EXAONE 4.0 model variants for causal language modeling."""

    EXAONE_4_0_32B_AWQ = "32B_Awq"


class ModelLoader(ForgeModel):
    """EXAONE 4.0 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EXAONE_4_0_32B_AWQ: LLMModelConfig(
            pretrained_model_name="LGAI-EXAONE/EXAONE-4.0-32B-AWQ",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EXAONE_4_0_32B_AWQ

    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="EXAONE 4.0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # AWQ variant: load on CPU and remove quantization_config
        # so weights are loaded as plain tensors.
        model_kwargs["device_map"] = "cpu"
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        return inputs
