# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechBrain MetricGAN+ Voicebank model loader for speech enhancement.
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
    """Available MetricGAN+ Voicebank model variants."""

    VOICEBANK = "Voicebank"


class MetricGANPlusModel(torch.nn.Module):
    """Wrapper module for the SpeechBrain MetricGAN+ enhancement model."""

    def __init__(self, enhance_model):
        super().__init__()
        self.enhance_model = enhance_model.mods.enhance_model

    def forward(self, noisy_features):
        return self.enhance_model(noisy_features)


class ModelLoader(ForgeModel):
    """SpeechBrain MetricGAN+ Voicebank model loader."""

    _VARIANTS = {
        ModelVariant.VOICEBANK: ModelConfig(
            pretrained_model_name="speechbrain/metricgan-plus-voicebank",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOICEBANK

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MetricGANPlusVoicebank",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SpeechBrain MetricGAN+ speech enhancement model."""
        from speechbrain.inference.enhancement import SpectralMaskEnhancement

        enhance_model = SpectralMaskEnhancement.from_hparams(
            source=self._variant_config.pretrained_model_name, **kwargs
        )

        model = MetricGANPlusModel(enhance_model)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate synthetic spectral features matching MetricGAN+ input shape.

        The enhance_model expects log spectral magnitude features of shape
        (batch, time_steps, frequency_bins).
        """
        # Approximate shape from 1s of 16kHz audio after STFT: (batch, ~101 frames, 257 freq bins)
        features = torch.randn(1, 101, 257)

        if dtype_override is not None:
            features = features.to(dtype_override)

        return [features]
