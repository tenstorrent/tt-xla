#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 WanGP model loader implementation.

Loads single-file safetensors VAE from wan-world/Wan2.2, a community repackage
of Wan 2.2 models optimized for WanGP inference.

Available variants:
- WAN22_VAE: Wan 2.2 VAE (z_dim=16, 3-channel RGB, fp32)
- WAN22_VAE_BF16: Wan 2.2 VAE (z_dim=16, 3-channel RGB, bf16)
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

REPO_ID = "wan-world/Wan2.2"

# Wan 2.2 VAE uses 16 latent channels (z_dim=16)
LATENT_CHANNELS = 16

# Small test dimensions for VAE inputs
# Wan VAE compression: 4x temporal, 8x spatial
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 2  # temporal latent frames

# VAE file paths within the wan-world/Wan2.2 repo
_VAE_FILES = {
    "fp32": "Wan2.2_VAE.safetensors",
    "bf16": "Wan2.2_VAE_bf16.safetensors",
}

# Config source for loading VAE architecture
_VAE_CONFIG = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class ModelVariant(StrEnum):
    """Available wan-world/Wan2.2 model variants."""

    WAN22_VAE = "2.2_VAE"
    WAN22_VAE_BF16 = "2.2_VAE_BF16"


class ModelLoader(ForgeModel):
    """wan-world/Wan2.2 model loader using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.WAN22_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.WAN22_VAE_BF16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_WORLD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLWan:
        """Load VAE from single-file safetensors."""
        if self._variant == ModelVariant.WAN22_VAE_BF16:
            filename = _VAE_FILES["bf16"]
        else:
            filename = _VAE_FILES["fp32"]

        vae_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
        )

        self._vae = AutoencoderKLWan.from_single_file(
            vae_path,
            config=_VAE_CONFIG,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Wan 2.2 VAE model.

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
