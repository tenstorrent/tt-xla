# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wav2Small model loader implementation for audio classification (speech emotion recognition).
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
    """Available Wav2Small audio classification model variants."""

    WAV2SMALL = "Wav2Small"


class ModelLoader(ForgeModel):
    """Wav2Small model loader implementation for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.WAV2SMALL: ModelConfig(
            pretrained_model_name="audeering/wav2small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WAV2SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wav2Small",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from .src.model import Wav2Small, Wav2SmallConfig

        config = Wav2SmallConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Wav2Small.from_pretrained(
            self._variant_config.pretrained_model_name,
            config=config,
            **model_kwargs,
        )

        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np
        import torch

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        signal = torch.from_numpy(audio_array)[None, :]

        if dtype_override is not None:
            signal = signal.to(dtype_override)

        return signal
