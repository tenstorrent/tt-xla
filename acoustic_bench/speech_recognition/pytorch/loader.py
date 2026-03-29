# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Acoustic Bench SALM model loader implementation for speech recognition (ASR) using PyTorch.
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
    """Available Acoustic Bench PyTorch speech recognition model variants."""

    ACOUSTIC_BENCH = "Acoustic_Bench"


class ModelLoader(ForgeModel):
    """Acoustic Bench SALM model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.ACOUSTIC_BENCH: ModelConfig(
            pretrained_model_name="KarthikSivaramaKrishnan/acoustic-bench",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ACOUSTIC_BENCH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Acoustic_Bench",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from nemo.collections.speechlm2.models import SALM

        model = SALM.restore_from(
            self._variant_config.pretrained_model_name,
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
