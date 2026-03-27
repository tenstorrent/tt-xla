# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DialoGPT model loader implementations for conversational text generation.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DialoGPT model variants."""

    DIALOGPT_SMALL = "Small"
    DIALOGPT_MEDIUM = "Medium"
    DIALOGPT_LARGE = "Large"


class ModelLoader(ForgeModel):
    """DialoGPT loader for conversational causal language modeling."""

    _VARIANTS = {
        ModelVariant.DIALOGPT_SMALL: LLMModelConfig(
            pretrained_model_name="microsoft/DialoGPT-small",
            max_length=256,
        ),
        ModelVariant.DIALOGPT_MEDIUM: LLMModelConfig(
            pretrained_model_name="microsoft/DialoGPT-medium",
            max_length=256,
        ),
        ModelVariant.DIALOGPT_LARGE: LLMModelConfig(
            pretrained_model_name="microsoft/DialoGPT-large",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIALOGPT_SMALL

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
            model="DialoGPT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        config = GPT2Config.from_pretrained(model_name)
        config_dict = config.to_dict()
        config_dict["use_cache"] = True
        if dtype_override is not None:
            config_dict["torch_dtype"] = dtype_override
        if self.num_layers is not None:
            config_dict["num_hidden_layers"] = self.num_layers
        config = GPT2Config(**config_dict)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, **kwargs
        )
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        vocab_size = GPT2Config.from_pretrained(
            self._variant_config.pretrained_model_name
        ).vocab_size

        input_ids = torch.cat(
            [
                torch.randint(1, vocab_size, (1, 255)),
                torch.zeros(1, 1, dtype=torch.int64),
            ],
            dim=-1,
        ).to(torch.int64)

        return {"input_ids": input_ids}

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
