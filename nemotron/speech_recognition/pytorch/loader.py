# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron Speech Streaming model loader implementation for speech recognition (ASR) using PyTorch.
"""

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Nemotron Speech Streaming model variants."""

    NEMOTRON_SPEECH_STREAMING_EN_0_6B = "Speech_Streaming_EN_0.6B"


class ModelLoader(ForgeModel):
    """Nemotron Speech Streaming model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_SPEECH_STREAMING_EN_0_6B: ModelConfig(
            pretrained_model_name="nvidia/nemotron-speech-streaming-en-0.6b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_SPEECH_STREAMING_EN_0_6B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nemotron",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        input_signal = torch.tensor(audio_array).unsqueeze(0)
        input_signal_length = torch.tensor([len(audio_array)])

        if dtype_override is not None:
            input_signal = input_signal.to(dtype_override)

        return {
            "input_signal": input_signal,
            "input_signal_length": input_signal_length,
        }
