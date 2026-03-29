# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MedVAE model loader implementation.

Loads stanfordmimi/MedVAE variational autoencoders for medical image
compression. These models encode high-resolution medical images into compact
latent representations and can decode them back.

Available variants:
- MEDVAE_4_1_2D: 4x spatial compression, 1-channel grayscale (16x total)
- MEDVAE_4_3_2D: 4x spatial compression, 3-channel RGB (16x total)
- MEDVAE_8_1_2D: 8x spatial compression, 1-channel grayscale (64x total)
"""

from typing import Any, Optional

import torch
from medvae import MVAE  # type: ignore[import]

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

REPO_ID = "stanfordmimi/MedVAE"

# Small test spatial dimensions (must be divisible by compression factor)
IMAGE_SIZE = 64


class ModelVariant(StrEnum):
    """Available MedVAE model variants."""

    MEDVAE_4_1_2D = "medvae_4_1_2d"
    MEDVAE_4_3_2D = "medvae_4_3_2d"
    MEDVAE_8_1_2D = "medvae_8_1_2d"


# Map variants to their input channel counts
_VARIANT_CHANNELS = {
    ModelVariant.MEDVAE_4_1_2D: 1,
    ModelVariant.MEDVAE_4_3_2D: 3,
    ModelVariant.MEDVAE_8_1_2D: 1,
}

# Map variants to their modality for MVAE constructor
_VARIANT_MODALITY = {
    ModelVariant.MEDVAE_4_1_2D: "xray",
    ModelVariant.MEDVAE_4_3_2D: "xray",
    ModelVariant.MEDVAE_8_1_2D: "xray",
}


class ModelLoader(ForgeModel):
    """MedVAE model loader for medical image VAEs."""

    _VARIANTS = {
        ModelVariant.MEDVAE_4_1_2D: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.MEDVAE_4_3_2D: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.MEDVAE_8_1_2D: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.MEDVAE_4_3_2D

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MedVAE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the MedVAE model.

        Returns:
            MVAE instance for medical image encoding/decoding.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._model is None:
            self._model = MVAE(
                model_name=self._variant.value,
                modality=_VARIANT_MODALITY[self._variant],
            )
            self._model.requires_grad_(False)
            self._model.eval()
            self._model = self._model.to(dtype=dtype)
        elif dtype_override is not None:
            self._model = self._model.to(dtype=dtype_override)
        return self._model

    def load_inputs(self, **kwargs) -> Any:
        """Prepare synthetic image inputs for the MedVAE model.

        Returns:
            Image tensor of shape [batch, channels, height, width].
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        channels = _VARIANT_CHANNELS[self._variant]
        return torch.randn(1, channels, IMAGE_SIZE, IMAGE_SIZE, dtype=dtype)
