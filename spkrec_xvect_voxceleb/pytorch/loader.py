# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechBrain X-Vector speaker recognition model loader for speaker embeddings.
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
    """Available SpeechBrain X-Vector model variants."""

    XVECT_VOXCELEB = "XVect_VoxCeleb"


class ModelLoader(ForgeModel):
    """SpeechBrain X-Vector speaker recognition model loader."""

    _VARIANTS = {
        ModelVariant.XVECT_VOXCELEB: ModelConfig(
            pretrained_model_name="speechbrain/spkrec-xvect-voxceleb",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XVECT_VOXCELEB

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SpeechBrainXVectVoxCeleb",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SpeechBrain X-Vector model."""
        import torchaudio

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []

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
        """Generate sample Fbank feature input for the X-Vector embedding model.

        Returns pre-computed features of shape (batch, time_steps, n_mels)
        equivalent to 1 second of 16kHz audio processed through Fbank features.
        """
        # X-Vector embedding model expects (batch, time_steps, n_mels)
        # 1 second of 16kHz audio produces ~101 frames with 80 Mel filters
        features = torch.randn(1, 101, 80)

        if dtype_override is not None:
            features = features.to(dtype_override)

        return [features]
