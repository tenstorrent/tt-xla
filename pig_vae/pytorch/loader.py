#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pig VAE model loader implementation.

Loads the calcuis/pig-vae GGUF-format VAE models using diffusers'
GGUF quantization support. These are standard VAE architectures
(e.g. Stable Diffusion AutoencoderKL) distributed in GGUF format
for efficient storage and inference.

Available variants:
- SD_VAE_FP16: Stable Diffusion VAE in fp16 GGUF format
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKL, GGUFQuantizationConfig  # type: ignore[import]
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

REPO_ID = "calcuis/pig-vae"

# Variant-specific GGUF filenames
SD_VAE_FILENAME = "pig_sd_vae_fp32-f16.gguf"

# SD VAE config source for architecture definition
SD_VAE_CONFIG = "stabilityai/sd-vae-ft-mse"

# SD VAE latent space: 4 channels, 8x spatial compression
LATENT_CHANNELS = 4
LATENT_HEIGHT = 64
LATENT_WIDTH = 64


class ModelVariant(StrEnum):
    """Available Pig VAE model variants."""

    SD_VAE_FP16 = "sd_vae_fp16"


class ModelLoader(ForgeModel):
    """Pig VAE model loader for GGUF-format VAE models."""

    _VARIANTS = {
        ModelVariant.SD_VAE_FP16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SD_VAE_FP16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PIG_VAE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Pig VAE model from GGUF format.

        Returns:
            AutoencoderKL instance loaded from GGUF checkpoint.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            gguf_path = hf_hub_download(REPO_ID, SD_VAE_FILENAME)
            self._vae = AutoencoderKL.from_single_file(
                gguf_path,
                config=SD_VAE_CONFIG,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            self._vae.eval()
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(self, **kwargs) -> Any:
        """Prepare latent inputs for the VAE decoder.

        Returns:
            Latent tensor of shape [batch, 4, 64, 64].
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        return torch.randn(
            1,
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            dtype=dtype,
        )
