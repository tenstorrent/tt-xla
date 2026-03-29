#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 T2V A14B Lightning model loader implementation.

magespace/Wan2.2-T2V-A14B-Lightning-Diffusers is a distilled/accelerated
variant of the Wan 2.2 Text-to-Video 14B model, designed for fast inference
with fewer diffusion steps. It uses a dual WanTransformer3DModel architecture,
AutoencoderKLWan VAE, and UMT5 text encoder.

Supports:
- Full pipeline loading (subfolder=None)
- VAE component loading (subfolder="vae") for encoder/decoder testing

Available variants:
- WAN22_T2V_A14B_LIGHTNING: Full pipeline
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLWan, WanPipeline  # type: ignore[import]

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

REPO_ID = "magespace/Wan2.2-T2V-A14B-Lightning-Diffusers"

SUPPORTED_SUBFOLDERS = {"vae"}

# Wan 2.x VAE uses 16 latent channels (z_dim=16)
LATENT_CHANNELS = 16

# Small test dimensions for VAE inputs
# Wan VAE compression: 4x temporal, 8x spatial
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 2  # temporal latent frames


class ModelVariant(StrEnum):
    """Available Wan 2.2 T2V Lightning model variants."""

    WAN22_T2V_A14B_LIGHTNING = "2.2_T2V_A14B_Lightning"


class ModelLoader(ForgeModel):
    """Wan 2.2 T2V A14B Lightning model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_T2V_A14B_LIGHTNING: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_T2V_A14B_LIGHTNING

    DEFAULT_PROMPT = (
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, subfolder: Optional[str] = None
    ):
        super().__init__(variant)
        if subfolder is not None and subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder
        self._vae = None
        self.pipeline: Optional[WanPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_22_T2V_LIGHTNING",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLWan:
        """Load VAE component for encoder/decoder testing."""
        self._vae = AutoencoderKLWan.from_pretrained(
            REPO_ID,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae.eval()
        return self._vae

    def _load_pipeline(self, dtype: torch.dtype) -> WanPipeline:
        """Load the full WanPipeline.

        The VAE is loaded in float32 for numerical stability while
        the rest of the pipeline uses the specified dtype.
        """
        vae = AutoencoderKLWan.from_pretrained(
            REPO_ID,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        self.pipeline = WanPipeline.from_pretrained(
            REPO_ID,
            vae=vae,
            torch_dtype=dtype,
        )
        return self.pipeline

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Wan pipeline or VAE component.

        Returns:
            WanPipeline or AutoencoderKLWan depending on subfolder.
        """
        if self._subfolder == "vae":
            dtype = dtype_override if dtype_override is not None else torch.float32
            if self._vae is None:
                return self._load_vae(dtype)
            if dtype_override is not None:
                self._vae = self._vae.to(dtype=dtype_override)
            return self._vae

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipeline is None:
            return self._load_pipeline(dtype)
        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)
        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for the model or component.

        For VAE subfolder, pass vae_type="decoder" or vae_type="encoder".
        For full pipeline, returns a prompt dict.
        """
        if self._subfolder == "vae":
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

        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
