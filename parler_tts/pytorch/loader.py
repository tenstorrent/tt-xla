# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Parler-TTS model loader implementation for text-to-speech tasks.
"""
import torch.nn as nn
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

    Exposes a clean forward pass that takes tokenized description and prompt
    and produces audio waveform output.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, prompt_input_ids):
        generation = self.model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
        )
        return generation


class ModelVariant(StrEnum):
    """Available Parler-TTS model variants."""

    PARLER_TTS_MINI_V1 = "parler-tts-mini-v1"


class ModelLoader(ForgeModel):
    """Parler-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.PARLER_TTS_MINI_V1: ModelConfig(
            pretrained_model_name="parler-tts/parler-tts-mini-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PARLER_TTS_MINI_V1

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
        from transformers import AutoTokenizer

        model_name = self._variant_config.pretrained_model_name
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        wrapper = ParlerTTSWrapper(model)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None):
        description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
        prompt = "Hello, this is a test of the Parler TTS model."

        input_ids = self.tokenizer(description, return_tensors="pt").input_ids
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        return input_ids, prompt_input_ids
