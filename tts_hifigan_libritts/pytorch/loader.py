# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechBrain HiFi-GAN LibriTTS vocoder model loader implementation.

Converts mel spectrograms to audio waveforms at 22050 Hz sample rate.
"""
import torch
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
    """Available SpeechBrain HiFi-GAN LibriTTS model variants."""

    HIFIGAN_22050HZ = "hifigan_22050hz"


class ModelLoader(ForgeModel):
    """SpeechBrain HiFi-GAN LibriTTS vocoder model loader implementation."""

    _VARIANTS = {
        ModelVariant.HIFIGAN_22050HZ: ModelConfig(
            pretrained_model_name="speechbrain/tts-hifigan-libritts-22050Hz",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HIFIGAN_22050HZ

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SpeechBrainHiFiGANLibriTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SpeechBrain HiFi-GAN vocoder generator module."""
        from speechbrain.inference.vocoders import HIFIGAN

        hifi_gan = HIFIGAN.from_hparams(
            source=self._variant_config.pretrained_model_name,
            savedir="pretrained_models/tts-hifigan-libritts-22050Hz",
        )
        model = hifi_gan.hparams.generator
        model.remove_weight_norm()
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample mel spectrogram input for the HiFi-GAN vocoder.

        Returns a mel spectrogram tensor of shape (batch, mel_bins, time_steps).
        The generator expects 80 mel bins and produces waveforms at 256x upsampling.
        """
        # Shape: (batch, mel_bins, time_steps)
        mel_spectrogram = torch.randn(1, 80, 100)

        if dtype_override is not None:
            mel_spectrogram = mel_spectrogram.to(dtype_override)

        return (mel_spectrogram,)
