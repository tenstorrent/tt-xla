# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KittenTTS Nano model loader implementation for text-to-speech tasks.
"""
import torch
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


class KittenTTSWrapper(nn.Module):
    """Wrapper around KittenTTS to expose generate as forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        audio = self.model.generate(text)
        return torch.tensor(audio)


class ModelVariant(StrEnum):
    """Available KittenTTS model variants."""

    KITTEN_TTS_NANO_0_1 = "nano-0.1"


class ModelLoader(ForgeModel):
    """KittenTTS Nano model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.KITTEN_TTS_NANO_0_1: ModelConfig(
            pretrained_model_name="KittenML/kitten-tts-nano-0.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KITTEN_TTS_NANO_0_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tts_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KittenTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from kittentts import KittenTTS

        self.tts_model = KittenTTS(self._variant_config.pretrained_model_name)
        model = KittenTTSWrapper(self.tts_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        text = "Hello, this is a test of text to speech."
        return (text,)
