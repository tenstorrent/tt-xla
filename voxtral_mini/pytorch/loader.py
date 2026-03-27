# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Voxtral Mini model loader implementation for real-time speech recognition (ASR).
"""

import numpy as np
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


class ModelVariant(StrEnum):
    """Available Voxtral Mini model variants."""

    VOXTRAL_MINI_4B_REALTIME = "Voxtral_Mini_4B_Realtime"


class ModelLoader(ForgeModel):
    """Voxtral Mini model loader implementation for real-time ASR."""

    _VARIANTS = {
        ModelVariant.VOXTRAL_MINI_4B_REALTIME: ModelConfig(
            pretrained_model_name="mistralai/Voxtral-Mini-4B-Realtime-2602",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOXTRAL_MINI_4B_REALTIME

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Voxtral-Mini",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import VoxtralRealtimeForConditionalGeneration

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def _load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def load_inputs(self, dtype_override=None):
        if self._processor is None:
            self._load_processor()

        # Generate a synthetic 1-second audio waveform at the expected sampling rate
        sampling_rate = self._processor.feature_extractor.sampling_rate
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(audio_array, return_tensors="pt")

        return inputs
