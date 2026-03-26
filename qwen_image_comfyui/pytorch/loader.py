# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image ComfyUI Repackaged model loader implementation.

Loads the VAE component from Comfy-Org/Qwen-Image_ComfyUI via the upstream
Qwen/Qwen-Image-2512 diffusers config. Supports encoder/decoder testing.

Available variants:
- QWEN_IMAGE_VAE: Qwen-Image VAE (z_dim=16, 3-channel RGB)
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLQwenImage

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

REPO_ID = "Comfy-Org/Qwen-Image_ComfyUI"
UPSTREAM_REPO = "Qwen/Qwen-Image-2512"

# z_dim from model config
LATENT_CHANNELS = 16
LATENT_FRAMES = 1
LATENT_HEIGHT = 8
LATENT_WIDTH = 8


class ModelVariant(StrEnum):
    """Available Qwen-Image ComfyUI model variants."""

    QWEN_IMAGE_VAE = "VAE"


class ModelLoader(ForgeModel):
    """Qwen-Image ComfyUI model loader for VAE component."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLQwenImage:
        """Load VAE from the upstream Qwen-Image-2512 repository."""
        self._vae = AutoencoderKLQwenImage.from_pretrained(
            UPSTREAM_REPO,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Qwen-Image VAE model.

        Returns:
            AutoencoderKLQwenImage instance.
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
            # [batch, channels, frames, height, width]
            return torch.randn(
                1,
                LATENT_CHANNELS,
                LATENT_FRAMES,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            )
        elif vae_type == "encoder":
            # [batch, channels, frames, height, width]
            return torch.randn(
                1, 3, LATENT_FRAMES, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype
            )
        else:
            raise ValueError(
                f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
            )
