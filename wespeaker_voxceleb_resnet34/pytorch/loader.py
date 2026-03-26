# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WeSpeaker VoxCeleb ResNet34-LM model loader implementation for speaker embeddings.
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
    """Available WeSpeaker VoxCeleb ResNet34-LM model variants."""

    RESNET34_LM = "ResNet34_LM"


class ModelLoader(ForgeModel):
    """WeSpeaker VoxCeleb ResNet34-LM model loader implementation."""

    _VARIANTS = {
        ModelVariant.RESNET34_LM: ModelConfig(
            pretrained_model_name="pyannote/wespeaker-voxceleb-resnet34-LM",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RESNET34_LM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="WeSpeakerVoxCelebResNet34",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the WeSpeaker VoxCeleb ResNet34-LM model from pyannote.audio."""
        from pyannote.audio import Model

        model = Model.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample audio input for the model.

        Returns a 1-second mono waveform at 16kHz sample rate.
        """
        # Model expects (batch, channels, samples): 16kHz mono audio, 1 second
        waveform = torch.randn(1, 1, 16000)

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)

        return [waveform]
