# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Zero123++ v1.2 model loader implementation.

Zero123++ is a single-image-to-consistent-multi-view diffusion model.
Given one input image, it generates 6 novel views at fixed camera poses.

Available variants:
- UNET: UNet2DConditionModel from the Zero123++ pipeline
- VAE: AutoencoderKL from the Zero123++ pipeline
"""

from typing import Optional

import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel

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

REPO_ID = "sudo-ai/zero123plus-v1.2"

# Latent dimensions for sample inputs
LATENT_CHANNELS = 4
LATENT_HEIGHT = 12
LATENT_WIDTH = 12


class ModelVariant(StrEnum):
    """Available Zero123++ model variants."""

    UNET = "UNet"
    VAE = "VAE"


class ModelLoader(ForgeModel):
    """Zero123++ v1.2 model loader."""

    _VARIANTS = {
        ModelVariant.UNET: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.UNET

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Zero123Plus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.float16) -> DiffusionPipeline:
        """Load the Zero123++ pipeline."""
        if self._pipe is None:
            self._pipe = DiffusionPipeline.from_pretrained(
                REPO_ID,
                torch_dtype=dtype,
            )
            self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self._pipe.scheduler.config, timestep_spacing="trailing"
            )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the model component for the selected variant.

        For UNET: returns the UNet2DConditionModel.
        For VAE: returns the AutoencoderKL.
        """
        dtype = dtype_override or torch.float16
        pipe = self._load_pipeline(dtype)

        if self._variant == ModelVariant.VAE:
            return pipe.vae

        return pipe.unet

    def load_inputs(self, **kwargs):
        """Prepare sample inputs for the selected variant.

        For UNET: returns (latents, timestep, encoder_hidden_states).
        For VAE: returns latent tensor for decoding.
        """
        dtype = kwargs.get("dtype_override", torch.float16)

        if self._variant == ModelVariant.VAE:
            return torch.randn(
                1,
                LATENT_CHANNELS,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            )

        # UNet inputs: sample, timestep, encoder_hidden_states
        sample = torch.randn(
            1,
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            dtype=dtype,
        )
        timestep = torch.tensor([1.0], dtype=dtype)
        # SD 2.x text encoder has hidden_size=1024
        encoder_hidden_states = torch.randn(1, 77, 1024, dtype=dtype)

        return [sample, timestep, encoder_hidden_states]
