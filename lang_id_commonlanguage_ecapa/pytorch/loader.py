# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechBrain ECAPA-TDNN language identification model loader for audio classification.
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
    """Available SpeechBrain ECAPA-TDNN language identification model variants."""

    COMMONLANGUAGE_ECAPA = "CommonLanguage_ECAPA"


class ModelLoader(ForgeModel):
    """SpeechBrain ECAPA-TDNN language identification model loader."""

    _VARIANTS = {
        ModelVariant.COMMONLANGUAGE_ECAPA: ModelConfig(
            pretrained_model_name="speechbrain/lang-id-commonlanguage_ecapa",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COMMONLANGUAGE_ECAPA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SpeechBrainLangIdECAPA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SpeechBrain ECAPA-TDNN language identification model."""
        from speechbrain.inference.classifiers import EncoderClassifier

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
        """Generate sample Fbank feature input for the ECAPA-TDNN embedding model.

        Returns pre-computed features of shape (batch, time_steps, n_mels)
        equivalent to 1 second of 16kHz audio processed through Fbank features.
        """
        # ECAPA-TDNN embedding model expects (batch, time_steps, n_mels)
        # 1 second of 16kHz audio produces ~101 frames with 80 Mel filters
        features = torch.randn(1, 101, 80)

        if dtype_override is not None:
            features = features.to(dtype_override)

        return [features]
