# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Parler-TTS model loader implementation for text-to-speech tasks.
"""
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ParlerTTSWrapper(nn.Module):
    """Wrapper around ParlerTTSForConditionalGeneration.

    Exposes the encoder-decoder forward pass that takes tokenized description
    and decoder input IDs, returning logits for audio code prediction.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs.logits


class ModelVariant(StrEnum):
    """Available Parler-TTS model variants."""

    MINI_V0_1 = "mini_v0.1"


class ModelLoader(ForgeModel):
    """Parler-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MINI_V0_1: ModelConfig(
            pretrained_model_name="parler-tts/parler_tts_mini_v0.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINI_V0_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Parler-TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from parler_tts import ParlerTTSForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        full_model = ParlerTTSForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model = ParlerTTSWrapper(full_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        description = (
            "A female speaker with a slightly low-pitched voice delivers her words"
            " quite expressively, in a very confined sounding environment with clear"
            " audio quality. She speaks very fast."
        )
        prompt = "Hey, how are you doing today?"

        description_tokens = self.tokenizer(description, return_tensors="pt")
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")

        return (
            description_tokens["input_ids"],
            description_tokens["attention_mask"],
            prompt_tokens["input_ids"],
        )
