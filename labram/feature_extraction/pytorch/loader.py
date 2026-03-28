# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LaBraM (Large Brain Model) loader implementation for EEG feature extraction.
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
    """Available LaBraM feature extraction model variants."""

    PRETRAINED = "Pretrained"


class ModelLoader(ForgeModel):
    """LaBraM model loader implementation for EEG feature extraction."""

    _VARIANTS = {
        ModelVariant.PRETRAINED: ModelConfig(
            pretrained_model_name="braindecode/labram-pretrained",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PRETRAINED

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LaBraM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from braindecode.models import Labram

        model = Labram.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # LaBraM expects EEG data with 128 channels and 3000 time samples
        n_channels = 128
        n_times = 3000
        batch_size = 1

        eeg_data = np.random.randn(batch_size, n_channels, n_times).astype(np.float32)
        inputs = torch.from_numpy(eeg_data)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
