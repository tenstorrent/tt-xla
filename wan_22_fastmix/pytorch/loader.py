#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 FastMix model loader implementation.

Loads VAE from the Wan I2V pipeline for encoder/decoder testing.
Zuntan/Wan22-FastMix is an Image-to-Video (I2V) model based on Wan 2.2 14B,
optimized for fast 6-step inference using merged distillation LoRAs.

Available variants:
- WAN22_FASTMIX_VAE: Wan 2.2 VAE (z_dim=16, 3-channel RGB)
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLWan  # type: ignore[import]

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

REPO_ID = "Zuntan/Wan22-FastMix"

# VAE config source (Wan 2.2 FastMix uses the same VAE as Wan 2.1 I2V)
_VAE_CONFIG = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# Wan 2.x VAE uses 16 latent channels (z_dim=16)
LATENT_CHANNELS = 16

# Small test dimensions for VAE inputs
# Wan VAE compression: 4x temporal, 8x spatial
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 2  # temporal latent frames


class ModelVariant(StrEnum):
    """Available Wan 2.2 FastMix model variants."""

    WAN22_FASTMIX_VAE = "2.2_FastMix_VAE"


class ModelLoader(ForgeModel):
    """Wan 2.2 FastMix model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_FASTMIX_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_FASTMIX_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_22_FASTMIX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLWan:
        """Load VAE from the base Wan I2V config for encoder/decoder testing."""
        self._vae = AutoencoderKLWan.from_pretrained(
            _VAE_CONFIG,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Wan VAE model.

        Returns:
            AutoencoderKLWan instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            return self._load_vae(dtype)
        if dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the VAE.

        Pass vae_type="decoder" or vae_type="encoder" to select input type.
        Defaults to decoder inputs.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        vae_type = kwargs.get("vae_type", "decoder")

        if vae_type == "decoder":
            # [batch, channels, time, height, width]
            return torch.randn(
                1,
                LATENT_CHANNELS,
                LATENT_DEPTH,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            )
        elif vae_type == "encoder":
            # T must satisfy T = 1 + 4*N (Wan temporal constraint)
            num_frames = 1 + 4 * LATENT_DEPTH  # 9 frames
            return torch.randn(
                1, 3, num_frames, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype
            )
        else:
            raise ValueError(
                f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
            )
