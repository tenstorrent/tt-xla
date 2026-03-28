# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Zero123-XL model loader implementation.

Zero-1-to-3 XL is a diffusion model for novel view synthesis from a single image.
It generates new viewpoints of an object given a single input image and target
camera pose (elevation, azimuth, radius).

Available variants:
- BASE: ashawkey/zero123-xl-diffusers (Zero123Pipeline)
"""

from typing import Any, Optional

import torch
from diffusers import Zero123Pipeline
from diffusers.utils import load_image

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

REPO_ID = "ashawkey/zero123-xl-diffusers"


class ModelVariant(StrEnum):
    """Available Zero123-XL model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Zero123-XL model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Zero123-XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.float32) -> Zero123Pipeline:
        """Load the Zero123-XL pipeline."""
        self._pipe = Zero123Pipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the UNet from the Zero123-XL pipeline.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._pipe is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self._pipe.unet = self._pipe.unet.to(dtype_override)
        return self._pipe.unet

    def load_inputs(self, **kwargs) -> Any:
        """Prepare synthetic inputs for the Zero123-XL UNet.

        Returns:
            list: [latent_model_input, timestep, encoder_hidden_states]
        """
        dtype = kwargs.get("dtype_override", torch.float32)

        if self._pipe is None:
            self._load_pipeline(dtype)

        # UNet input dimensions
        num_channels = (
            self._pipe.unet.config.in_channels
        )  # typically 8 (4 latent + 4 image latent)
        height = 32
        width = 32

        # Latent input (noisy latents concatenated with image latents)
        latent_model_input = torch.randn(1, num_channels, height, width, dtype=dtype)

        # Timestep
        timestep = torch.tensor([500], dtype=torch.long)

        # Encoder hidden states from CLIP image encoder
        cross_attention_dim = self._pipe.unet.config.cross_attention_dim
        encoder_hidden_states = torch.randn(1, 1, cross_attention_dim, dtype=dtype)

        return [latent_model_input, timestep, encoder_hidden_states]
