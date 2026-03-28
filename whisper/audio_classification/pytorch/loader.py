# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Whisper model loader implementation for audio classification (speech emotion recognition).
"""

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
    """Available Whisper audio classification model variants."""

    SPEECH_EMOTION_RECOGNITION = "Speech_Emotion_Recognition"


class ModelLoader(ForgeModel):
    """Whisper model loader implementation for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.SPEECH_EMOTION_RECOGNITION: ModelConfig(
            pretrained_model_name="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPEECH_EMOTION_RECOGNITION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self, dtype_override=None):
        from transformers import AutoFeatureExtractor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._feature_extractor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForAudioClassification

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForAudioClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._feature_extractor is None:
            self._load_feature_extractor(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
