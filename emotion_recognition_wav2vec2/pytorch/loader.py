# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechBrain Emotion Recognition Wav2Vec2 IEMOCAP model loader for audio classification.
"""

from typing import Optional

import torch

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Emotion Recognition Wav2Vec2 model variants."""

    IEMOCAP = "IEMOCAP"


class EmotionRecognitionModel(torch.nn.Module):
    """Wrapper module for the SpeechBrain emotion recognition pipeline."""

    def __init__(self, classifier):
        super().__init__()
        self.wav2vec2 = classifier.mods.wav2vec2
        self.avg_pool = classifier.mods.avg_pool
        self.output_mlp = classifier.mods.output_mlp

    def forward(self, wavs, wav_lens):
        feats = self.wav2vec2(wavs)
        pooled = self.avg_pool(feats, wav_lens)
        logits = self.output_mlp(pooled)
        return logits


class ModelLoader(ForgeModel):
    """SpeechBrain Emotion Recognition Wav2Vec2 IEMOCAP model loader."""

    _VARIANTS = {
        ModelVariant.IEMOCAP: ModelConfig(
            pretrained_model_name="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IEMOCAP

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="EmotionRecognitionWav2Vec2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SpeechBrain emotion recognition model."""
        from speechbrain.inference.interfaces import foreign_class

        classifier = foreign_class(
            source=self._variant_config.pretrained_model_name,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            **kwargs,
        )

        model = EmotionRecognitionModel(classifier)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate synthetic 1-second audio waveform at 16kHz."""
        waveform = torch.randn(1, 16000)
        wav_lens = torch.tensor([1.0])

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)
            wav_lens = wav_lens.to(dtype_override)

        return [waveform, wav_lens]
