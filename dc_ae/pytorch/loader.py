# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DC-AE (Deep Compression Autoencoder) model loader implementation.

Loads the mit-han-lab DC-AE autoencoder from HuggingFace using the diffusers
AutoencoderDC class. This model provides efficient high-resolution image
compression with a 32x spatial compression factor.

Available variants:
- F32C32_SANA_1_1: dc-ae-f32c32-sana-1.1-diffusers (32x compression, 32 latent channels)
"""

from typing import Optional

import torch
from diffusers import AutoencoderDC

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

REPO_ID = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"

# DC-AE f32c32: 32x spatial compression, 32 latent channels
LATENT_CHANNELS = 32
SPATIAL_COMPRESSION = 32

# Test input dimensions (512x512 image -> 16x16 latent)
INPUT_HEIGHT = 512
INPUT_WIDTH = 512


class ModelVariant(StrEnum):
    """Available DC-AE model variants."""

    F32C32_SANA_1_1 = "f32c32_sana_1.1"


class ModelLoader(ForgeModel):
    """DC-AE (Deep Compression Autoencoder) model loader."""

    _VARIANTS = {
        ModelVariant.F32C32_SANA_1_1: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.F32C32_SANA_1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DC-AE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the DC-AE model.

        Returns:
            AutoencoderDC instance for image encoding/decoding.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._model is None:
            self._model = AutoencoderDC.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
            )
            self._model.eval()
        elif dtype_override is not None:
            self._model = self._model.to(dtype=dtype_override)
        return self._model

    def load_inputs(self, **kwargs):
        """Prepare sample image input for the autoencoder.

        Returns:
            Tensor of shape [batch, 3, 512, 512] representing an RGB image.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)
        return torch.randn(
            batch_size,
            3,
            INPUT_HEIGHT,
            INPUT_WIDTH,
            dtype=dtype,
        )
