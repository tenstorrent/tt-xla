# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CBraMod pre-trained EEG foundation model loader implementation.
"""

import torch
from typing import Optional

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
    """Available CBraMod model variants."""

    PRETRAINED = "Pretrained"


class ModelLoader(ForgeModel):
    """CBraMod pre-trained EEG foundation model loader.

    CBraMod (Criss-Cross Brain Foundation Model) is a foundation model
    for EEG decoding using criss-cross spatial and temporal attention.
    Pre-trained on the Temple University Hospital EEG Corpus (TUEG).
    """

    _VARIANTS = {
        ModelVariant.PRETRAINED: ModelConfig(
            pretrained_model_name="braindecode/cbramod-pretrained",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PRETRAINED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CBraMod",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CBraMod pre-trained encoder.

        Returns:
            torch.nn.Module: CBraMod model instance configured for encoder output.
        """
        from braindecode.models import CBraMod

        model = CBraMod.from_pretrained(
            self._variant_config.pretrained_model_name,
            return_encoder_output=True,
        )

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample EEG inputs for the CBraMod model.

        Returns:
            torch.Tensor: Synthetic EEG tensor of shape (batch, n_channels, n_times).
        """
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)

        # 16 EEG channels, 2000 time steps (10 seconds at 200 Hz)
        # n_times must be divisible by patch_size (200)
        n_channels = 16
        n_times = 2000
        inputs = torch.randn(1, n_channels, n_times, dtype=dtype)

        return inputs
