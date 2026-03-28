#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 FastMix model loader implementation.

Loads single-file safetensors diffusion transformer variants from Zuntan/Wan22-FastMix.
This is an Image-to-Video (I2V) model based on Wan 2.2 14B, optimized for fast
6-step inference using merged distillation LoRAs.

The model uses a two-stage pipeline with separate HighNoise and LowNoise weights.

Available variants:
- WAN22_FASTMIX_HIGH_NOISE: HighNoise stage transformer (steps 0-3)
- WAN22_FASTMIX_LOW_NOISE: LowNoise stage transformer (steps 3-6)
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLWan  # type: ignore[import]
from huggingface_hub import hf_hub_download  # type: ignore[import]

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

# Wan 2.x VAE uses 16 latent channels (z_dim=16)
LATENT_CHANNELS = 16

# Small test dimensions for VAE inputs
# Wan VAE compression: 4x temporal, 8x spatial
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 2  # temporal latent frames

# Transformer file paths within the repo (fp8 variants for manageable size)
_TRANSFORMER_FILES = {
    "high_noise": "Wan22-I2V-FastMix_v10-H-fp8_e4m3fn.safetensors",
    "low_noise": "Wan22-I2V-FastMix_v10-L-fp8_e4m3fn.safetensors",
}

# Config source for loading transformer architecture
_TRANSFORMER_CONFIG = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.2 FastMix model variants."""

    WAN22_FASTMIX_HIGH_NOISE = "2.2_FastMix_HighNoise"
    WAN22_FASTMIX_LOW_NOISE = "2.2_FastMix_LowNoise"


class ModelLoader(ForgeModel):
    """Wan 2.2 FastMix model loader using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.WAN22_FASTMIX_HIGH_NOISE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.WAN22_FASTMIX_LOW_NOISE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_FASTMIX_HIGH_NOISE

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

    def _get_transformer_filename(self) -> str:
        """Get the safetensors filename for the current variant."""
        if self._variant == ModelVariant.WAN22_FASTMIX_LOW_NOISE:
            return _TRANSFORMER_FILES["low_noise"]
        return _TRANSFORMER_FILES["high_noise"]

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLWan:
        """Load VAE from the base Wan I2V config for encoder/decoder testing."""
        self._vae = AutoencoderKLWan.from_pretrained(
            _TRANSFORMER_CONFIG,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Wan 2.2 FastMix VAE model.

        Returns:
            AutoencoderKLWan instance for encoder/decoder testing.
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
