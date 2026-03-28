# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Silero VAD model loader implementation for voice activity detection.

Loads the Silero Voice Activity Detection model via torch.hub for
determining whether speech is present in an audio stream.
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
    """Available Silero VAD model variants."""

    V5 = "V5"


class ModelLoader(ForgeModel):
    """Silero VAD model loader implementation for voice activity detection."""

    _VARIANTS = {
        ModelVariant.V5: ModelConfig(
            pretrained_model_name="snakers4/silero-vad",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Silero_VAD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Silero VAD model via torch.hub."""
        model, _ = torch.hub.load(
            repo_or_dir=self._variant_config.pretrained_model_name,
            model="silero_vad",
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the Silero VAD model.

        Generates a 1-second mono audio waveform at 16kHz.
        The model expects a 1D tensor of audio samples.
        """
        dtype = dtype_override or torch.float32
        # 1 second of mono audio at 16kHz
        waveform = torch.randn(16000, dtype=dtype)
        return [waveform]
