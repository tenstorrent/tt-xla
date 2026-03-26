# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechBrain ECAPA-TDNN speaker recognition model loader for speaker embeddings.
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
    """Available SpeechBrain ECAPA-TDNN model variants."""

    ECAPA_VOXCELEB = "ECAPA_VoxCeleb"


class ModelLoader(ForgeModel):
    """SpeechBrain ECAPA-TDNN speaker recognition model loader."""

    _VARIANTS = {
        ModelVariant.ECAPA_VOXCELEB: ModelConfig(
            pretrained_model_name="speechbrain/spkrec-ecapa-voxceleb",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ECAPA_VOXCELEB

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SpeechBrainECAPAVoxCeleb",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SpeechBrain ECAPA-TDNN model."""
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source=self._variant_config.pretrained_model_name, **kwargs
        )
        model = classifier.mods.embedding_model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample audio input for the model.

        Returns a 1-second mono waveform at 16kHz sample rate.
        """
        # ECAPA-TDNN expects (batch, samples): 16kHz mono audio, 1 second
        waveform = torch.randn(1, 16000)

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)

        return [waveform]
