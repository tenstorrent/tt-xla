# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kanade disentangled speech tokenizer model loader implementation.

Kanade encodes audio waveforms into discrete content tokens and a global
speaker embedding, then decodes back to mel spectrograms.
"""

import torch
import numpy as np
from kanade_tokenizer import KanadeModel
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
    """Available Kanade model variants."""

    KANADE_25HZ_CLEAN = "Kanade 25Hz Clean"


class ModelLoader(ForgeModel):
    """Kanade speech tokenizer model loader implementation."""

    _VARIANTS = {
        ModelVariant.KANADE_25HZ_CLEAN: ModelConfig(
            pretrained_model_name="frothywater/kanade-25hz-clean",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KANADE_25HZ_CLEAN

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Kanade",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kanade model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = KanadeModel.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the Kanade model.

        Kanade expects a raw audio waveform at 24kHz sample rate
        with shape (samples,).
        """
        # Generate a synthetic 1-second audio waveform at 24kHz
        sampling_rate = 24000
        duration_seconds = 1
        audio = torch.from_numpy(
            np.random.randn(sampling_rate * duration_seconds).astype(np.float32)
        )

        if dtype_override is not None:
            audio = audio.to(dtype_override)

        return audio
