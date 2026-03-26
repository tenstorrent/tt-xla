# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image ComfyUI Repackaged model loader implementation.

Loads single-file safetensors VAE from Comfy-Org/Qwen-Image_ComfyUI.
Supports VAE component loading for encoder/decoder testing.

Available variants:
- QWEN_IMAGE_VAE: Qwen-Image VAE (latent channels=32, 3-channel RGB)
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLQwenImage
from huggingface_hub import hf_hub_download

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

LATENT_CHANNELS = 32
LATENT_HEIGHT = 8
LATENT_WIDTH = 8

_VAE_FILES = {
    "default": "split_files/vae/qwen_image_vae.safetensors",
}

_VAE_CONFIGS = {
    "default": "Qwen/Qwen-Image-2512",
}


class ModelVariant(StrEnum):
    """Available Qwen-Image ComfyUI model variants."""

    QWEN_IMAGE_VAE = "VAE"


class ModelLoader(ForgeModel):
    """Qwen-Image ComfyUI model loader using single-file safetensors."""

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
        """Load VAE from single-file safetensors."""
        vae_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_VAE_FILES["default"],
        )

        self._vae = AutoencoderKLQwenImage.from_single_file(
            vae_path,
            config=_VAE_CONFIGS["default"],
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
            return torch.randn(
                1,
                LATENT_CHANNELS,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            )
        elif vae_type == "encoder":
            return torch.randn(1, 3, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
            )
