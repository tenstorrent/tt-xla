# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mochi VAE model loader for tt_forge_models."""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.vae_utils import normalize_latents


@dataclass
class MochiVAEConfig(ModelConfig):
    """Configuration for Mochi VAE variants."""

    source: ModelSource
    enable_tiling: bool = False
    tile_sample_min_height: int = 128
    tile_sample_min_width: int = 128
    tile_sample_stride_height: int = 128
    tile_sample_stride_width: int = 128


class ModelVariant(StrEnum):
    """Available Mochi VAE variants."""

    MOCHI_VAE_DECODER = "mochi_vae_decoder"
    MOCHI_VAE_DECODER_TILED = "mochi_vae_decoder_tiled"


class ModelLoader(ForgeModel):
    """
    Loader for Mochi VAE decoder model.

    Mochi is a video generation model with a VAE that compresses video frames
    into latent representations. The decoder takes latents [B, 12, t, h, w] and
    produces RGB video frames [B, 3, T, H, W] with:
    - 6x temporal expansion
    - 8x8 spatial expansion

    Variants:
    - MOCHI_VAE_DECODER: Non-tiled decoder (memory intensive)
    - MOCHI_VAE_DECODER_TILED: Tiled decoder (memory efficient)
    """

    _VARIANTS = {
        ModelVariant.MOCHI_VAE_DECODER: MochiVAEConfig(
            pretrained_model_name="genmo/mochi-1-preview",
            source=ModelSource.HUGGING_FACE,
            enable_tiling=False,
        ),
        ModelVariant.MOCHI_VAE_DECODER_TILED: MochiVAEConfig(
            pretrained_model_name="genmo/mochi-1-preview",
            source=ModelSource.HUGGING_FACE,
            enable_tiling=True,
            tile_sample_min_height=128,
            tile_sample_min_width=128,
            tile_sample_stride_height=128,
            tile_sample_stride_width=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOCHI_VAE_DECODER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="mochi_vae",
            variant=variant,
            group=ModelGroup.PRIORITY,
            task=ModelTask.MM_VIDEO_TTT,  # Video generation task
            source=cls._VARIANTS[variant].source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None) -> torch.nn.Module:
        """
        Load Mochi VAE decoder model.

        Args:
            dtype_override: Optional dtype override (e.g., torch.bfloat16)

        Returns:
            For tiled variant: Full VAE model with tiling enabled
            For non-tiled variant: VAE decoder module only
        """
        from diffusers import AutoencoderKLMochi

        config = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Load VAE from HuggingFace (subfolder='vae' gets ~362M params, not full 11B)
        vae = AutoencoderKLMochi.from_pretrained(
            config.pretrained_model_name, subfolder="vae", torch_dtype=dtype
        )

        if config.enable_tiling:
            # Enable tiling for memory efficiency
            vae.enable_tiling(
                tile_sample_min_height=config.tile_sample_min_height,
                tile_sample_min_width=config.tile_sample_min_width,
                tile_sample_stride_height=config.tile_sample_stride_height,
                tile_sample_stride_width=config.tile_sample_stride_width,
            )
            # Keep all temporal frames
            vae.drop_last_temporal_frames = False
            model = vae
        else:
            # Use decoder module directly for non-tiled execution
            model = vae.decoder

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, **kwargs) -> torch.Tensor:
        """
        Load sample inputs for Mochi VAE decoder.

        Returns normalized latent tensor of shape [1, 12, 2, 16, 16] which
        will produce output shape [1, 3, 12, 128, 128] after decoding:
        - 12 latent channels
        - 2 frames -> 12 frames (6x temporal expansion)
        - 16x16 spatial -> 128x128 (8x8 spatial expansion)

        Args:
            dtype_override: Optional dtype override (e.g., torch.bfloat16)

        Returns:
            Normalized latent tensor ready for decoder input
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # [batch, channels, time, height, width]
        latent = torch.randn(1, 12, 2, 16, 16, dtype=dtype)

        latent_normalized = normalize_latents(latent, dtype=dtype)

        return latent_normalized

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        """
        Unpack model output to extract tensor.

        For tiled execution, output is an object with .sample attribute.
        For non-tiled execution, output is tuple (output, conv_cache) where output is a tensor.

        Args:
            output: Model forward pass output

        Returns:
            Output tensor of shape [B, 3, T, H, W]
        """
        if hasattr(output, "sample"):
            # Tiled decoder returns object with .sample attribute
            return output.sample
        else:
            # Non-tiled decoder returns tensor directly
            if isinstance(output, tuple):
                return output[0]
            return output
