# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SONICS SpecTTTra model loader implementation for audio classification (synthetic song detection).
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
    """Available SONICS SpecTTTra audio classification model variants."""

    ALPHA_120S = "Alpha_120s"


class ModelLoader(ForgeModel):
    """SONICS SpecTTTra model loader implementation for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.ALPHA_120S: ModelConfig(
            pretrained_model_name="awsaf49/sonics-spectttra-alpha-120s",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALPHA_120S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SpecTTTra",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from sonics import HFAudioClassifier

        model = HFAudioClassifier.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import torch

        # Generate a synthetic 1-second audio waveform at 16kHz
        # The model accepts raw waveforms and handles mel spectrogram conversion internally
        sampling_rate = 16000
        duration_seconds = 1
        waveform = torch.randn(1, sampling_rate * duration_seconds)

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)

        return {"audio": waveform}
