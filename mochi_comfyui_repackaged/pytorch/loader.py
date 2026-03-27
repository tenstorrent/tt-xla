#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mochi ComfyUI Repackaged model loader implementation.

Loads single-file safetensors VAE from Comfy-Org/mochi_preview_repackaged.
Supports VAE component loading for encoder/decoder testing.

Available variants:
- MOCHI_VAE: Mochi Preview VAE (12-channel latent, 3-channel RGB)
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLMochi  # type: ignore[import]
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

REPO_ID = "Comfy-Org/mochi_preview_repackaged"

# Mochi VAE uses 12 latent channels
LATENT_CHANNELS = 12

# Small test dimensions for VAE inputs
# Mochi VAE compression: 6x temporal, 8x spatial
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 2  # temporal latent frames


class ModelVariant(StrEnum):
    """Available Mochi ComfyUI Repackaged model variants."""

    MOCHI_VAE = "Preview_VAE"


class ModelLoader(ForgeModel):
    """Mochi ComfyUI Repackaged model loader using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.MOCHI_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.MOCHI_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MOCHI_COMFYUI_REPACKAGED",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLMochi:
        """Load VAE from single-file safetensors."""
        vae_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="split_files/vae/mochi_vae.safetensors",
        )

        self._vae = AutoencoderKLMochi.from_single_file(
            vae_path,
            config="genmo/mochi-1-preview",
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Mochi VAE model.

        Returns:
            AutoencoderKLMochi instance.
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
            # Mochi temporal compression is 6x
            num_frames = LATENT_DEPTH * 6  # 12 frames
            return torch.randn(
                1, 3, num_frames, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype
            )
        else:
            raise ValueError(
                f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
            )
