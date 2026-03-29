# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenSora VAE v1.2 model loader implementation.

Loads the hpcai-tech/OpenSora-VAE-v1.2 VideoAutoencoderPipeline, a two-stage
video VAE consisting of a spatial 2D VAE (from PixArt-alpha) and a temporal VAE.
The model encodes/decodes video frames into/from a 4-channel latent space.

Repository: https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2
"""

from typing import Any, Optional

import torch
from opensora.models.vae.vae import VideoAutoencoderPipeline  # type: ignore[import]

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

REPO_ID = "hpcai-tech/OpenSora-VAE-v1.2"

# The spatial VAE uses 4 latent channels with 8x spatial compression
LATENT_CHANNELS = 4
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 1  # single frame


class ModelVariant(StrEnum):
    """Available OpenSora VAE v1.2 model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """OpenSora VAE v1.2 model loader."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OpenSora_VAE_v1_2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the OpenSora VAE v1.2 model.

        Returns:
            VideoAutoencoderPipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            self._vae = VideoAutoencoderPipeline.from_pretrained(
                REPO_ID,
                torch_dtype=dtype,
            )
            self._vae.eval()
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(self, **kwargs) -> Any:
        """Prepare latent inputs for the VAE decoder.

        Returns:
            Latent tensor of shape [batch, 4, depth, height, width].
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        return torch.randn(
            1,
            LATENT_CHANNELS,
            LATENT_DEPTH,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            dtype=dtype,
        )
