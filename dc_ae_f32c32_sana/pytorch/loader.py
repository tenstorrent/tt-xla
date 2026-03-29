# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DC-AE (Deep Compression Autoencoder) model loader implementation.

Loads the mit-han-lab/dc-ae-f32c32-sana-1.0 autoencoder which provides
32x spatial compression with 32 latent channels. Used as the VAE component
in the SANA text-to-image diffusion pipeline.

Available variants:
- F32C32_SANA_1_0: dc-ae-f32c32-sana-1.0 (32x compression, 32 channels)
"""

from typing import Any, Optional

import torch
from efficientvit.ae_model_zoo import DCAE_HF

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

REPO_ID = "mit-han-lab/dc-ae-f32c32-sana-1.0"

# DC-AE f32c32: 32 latent channels, 32x spatial compression
LATENT_CHANNELS = 32
# Use small input dimensions for testing (must be divisible by 32)
INPUT_HEIGHT = 256
INPUT_WIDTH = 256


class ModelVariant(StrEnum):
    """Available DC-AE model variants."""

    F32C32_SANA_1_0 = "f32c32_sana_1.0"


class ModelLoader(ForgeModel):
    """DC-AE (Deep Compression Autoencoder) model loader."""

    _VARIANTS = {
        ModelVariant.F32C32_SANA_1_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.F32C32_SANA_1_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DC_AE_F32C32_SANA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the DC-AE model.

        Returns:
            DCAE_HF instance for image encoding/decoding.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._model is None:
            self._model = DCAE_HF.from_pretrained(REPO_ID)
            self._model.eval()
        if dtype_override is not None:
            self._model = self._model.to(dtype=dtype_override)
        return self._model

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample image input for the DC-AE model.

        Returns:
            Image tensor of shape [batch, 3, H, W] normalized to [-0.5, 0.5].
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        return torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH, dtype=dtype)
