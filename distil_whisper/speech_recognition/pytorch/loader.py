# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Distil-Whisper model loader implementation for speech recognition (ASR) using PyTorch.

Distil-small.en is a distilled version of OpenAI's Whisper small.en,
6x faster with 49% fewer parameters while performing within 1% WER.
"""

from typing import Optional

import numpy as np

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Distil-Whisper PyTorch speech recognition model variants."""

    DISTIL_SMALL_EN = "Distil_small_en"


class ModelLoader(ForgeModel):
    """Distil-Whisper model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.DISTIL_SMALL_EN: ModelConfig(
            pretrained_model_name="distil-whisper/distil-small.en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTIL_SMALL_EN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Distil_Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForSpeechSeq2Seq

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
