# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
z_image_turbo (Comfy-Org/z_image_turbo) model loader implementation.

Loads the VAE component from the Comfy-Org/z_image_turbo single-file safetensors.
Supports encoder/decoder testing of the autoencoder.

Available variants:
- Z_IMAGE_TURBO_VAE: z_image_turbo VAE autoencoder
"""

from typing import Any, Optional

import os

import torch
from diffusers import AutoencoderKL
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

REPO_ID = "Comfy-Org/z_image_turbo"

# VAE latent dimensions for testing
LATENT_CHANNELS = 16
LATENT_HEIGHT = 8
LATENT_WIDTH = 8


class ModelVariant(StrEnum):
    """Available z_image_turbo model variants."""

    Z_IMAGE_TURBO_VAE = "VAE"


class ModelLoader(ForgeModel):
    """z_image_turbo model loader using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.Z_IMAGE_TURBO_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Z_IMAGE_TURBO_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_IMAGE_TURBO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKL:
        """Load VAE from single-file safetensors."""
        vae_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="split_files/vae/ae.safetensors",
        )

        config_dir = os.path.join(os.path.dirname(__file__), "vae_config")
        self._vae = AutoencoderKL.from_single_file(
            vae_path,
            config=config_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the z_image_turbo VAE model.

        Returns:
            AutoencoderKL instance.
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
            # Latent input: [batch, channels, height, width]
            return torch.randn(
                1,
                LATENT_CHANNELS,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            )
        elif vae_type == "encoder":
            # Image input: [batch, 3, height, width]
            return torch.randn(1, 3, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
            )
