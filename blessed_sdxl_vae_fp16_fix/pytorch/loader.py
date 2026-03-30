# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
blessed_sdxl_vae_fp16_fix model loader implementation.

Loads the VAE (AutoencoderKL) from nubby/blessed-sdxl-vae-fp16-fix,
a fp16-safe SDXL VAE with adjusted contrast/brightness weights.

Available variants:
- BLESSED_SDXL_VAE_FP16_FIX: VAE autoencoder (nubby/blessed-sdxl-vae-fp16-fix)
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKL

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

REPO_ID = "nubby/blessed-sdxl-vae-fp16-fix"

# SDXL VAE latent dimensions for testing
LATENT_CHANNELS = 4
LATENT_HEIGHT = 8
LATENT_WIDTH = 8


class ModelVariant(StrEnum):
    """Available blessed_sdxl_vae_fp16_fix model variants."""

    BLESSED_SDXL_VAE_FP16_FIX = "blessed-sdxl-vae-fp16-fix"


class ModelLoader(ForgeModel):
    """blessed_sdxl_vae_fp16_fix model loader."""

    _VARIANTS = {
        ModelVariant.BLESSED_SDXL_VAE_FP16_FIX: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BLESSED_SDXL_VAE_FP16_FIX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BLESSED_SDXL_VAE_FP16_FIX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the AutoencoderKL VAE model."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained(
                REPO_ID,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            self._vae.eval()
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the VAE.

        Pass vae_type="decoder" (default) or vae_type="encoder".
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
