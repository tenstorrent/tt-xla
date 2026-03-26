#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan ComfyUI Repackaged model loader implementation.

Loads single-file safetensors VAE variants from Comfy-Org/Wan_2.2_ComfyUI_Repackaged.
Supports VAE component loading for encoder/decoder testing.

Available variants:
- WAN21_VAE: Wan 2.1 VAE (z_dim=16, 3-channel RGB)
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

REPO_ID = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# Wan 2.1 VAE uses 16 latent channels (z_dim=16)
LATENT_CHANNELS = 16

# Small test dimensions for VAE inputs
# Wan VAE compression: 4x temporal, 8x spatial
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 2  # temporal latent frames

# VAE file paths within the ComfyUI repackaged repo
_VAE_FILES = {
    "2.1": "split_files/vae/wan_2.1_vae.safetensors",
}

# Config sources for each VAE version
_VAE_CONFIGS = {
    "2.1": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
}


class ModelVariant(StrEnum):
    """Available Wan ComfyUI Repackaged model variants."""

    WAN21_VAE = "2.1_VAE"


class ModelLoader(ForgeModel):
    """Wan ComfyUI Repackaged model loader using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.WAN21_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLWan:
        """Load VAE from single-file safetensors."""
        vae_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_VAE_FILES["2.1"],
        )

        self._vae = AutoencoderKLWan.from_single_file(
            vae_path,
            config=_VAE_CONFIGS["2.1"],
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
