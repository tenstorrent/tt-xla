# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Difix model loader implementation.

Difix is a single-step image diffusion model that enhances rendered novel views
by removing artifacts from 3D reconstructions (NeRF/3DGS). It uses a
UNet2DConditionModel backbone with CLIP text conditioning.

Available variants:
- BASE: nvidia/difix (576x1024 image-to-image enhancement)
"""

import torch
from diffusers import DiffusionPipeline
from typing import Optional

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

# Input image dimensions expected by the model
IMAGE_HEIGHT = 576
IMAGE_WIDTH = 1024


class ModelVariant(StrEnum):
    """Available Difix model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Difix model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="nvidia/difix",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Difix",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load and cache the Difix pipeline."""
        pipe_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )
        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Difix UNet model.

        Returns:
            UNet2DConditionModel: The UNet backbone used for single-step denoising.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.unet = self.pipe.unet.to(dtype_override)

        return self.pipe.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Difix UNet.

        Returns:
            dict: Input tensors for the UNet forward pass.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32

        # Encode a text prompt using the CLIP text encoder
        prompt = "high quality, detailed"
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoder_hidden_states = self.pipe.text_encoder(text_inputs.input_ids)[0].to(
            dtype=dtype
        )

        # VAE spatial compression factor is typically 8
        vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        latent_height = IMAGE_HEIGHT // vae_scale_factor
        latent_width = IMAGE_WIDTH // vae_scale_factor
        num_channels = self.pipe.unet.config.in_channels

        # Latent sample input
        sample = torch.randn(
            batch_size,
            num_channels,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        # Single-step timestep
        timestep = torch.tensor([1], dtype=dtype)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
