# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2-dev-Turbo model loader implementation.

Loads the FLUX.2-dev base pipeline and applies the turbo distilled LoRA weights
from fal/FLUX.2-dev-Turbo for fast 8-step text-to-image generation.
"""

import torch
from diffusers import Flux2Pipeline
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

BASE_MODEL = "black-forest-labs/FLUX.2-dev"
LORA_REPO = "fal/FLUX.2-dev-Turbo"
LORA_WEIGHT_NAME = "flux.2-turbo-lora.safetensors"


class ModelVariant(StrEnum):
    """Available FLUX.2-dev-Turbo model variants."""

    DEV_TURBO = "Dev_Turbo"


class ModelLoader(ForgeModel):
    """FLUX.2-dev-Turbo model loader for fast text-to-image generation."""

    _VARIANTS = {
        ModelVariant.DEV_TURBO: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEV_TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[Flux2Pipeline] = None
        self.guidance_scale = 2.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.2-dev-Turbo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the FLUX.2-dev pipeline with turbo LoRA weights applied.

        Returns:
            The FLUX.2 transformer model with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.pipeline = Flux2Pipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the FLUX.2-dev-Turbo transformer.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.pipeline.transformer.config

        # Image dimensions
        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        # Prepare latents: VAE compresses by vae_scale_factor, then pack 2x2 patches
        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        # Create latent tensor (B, C, H, W) then pack to (B, H*W, C)
        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )

        # Prepare latent image IDs (B, H*W, 4)
        t = torch.arange(1)
        h = torch.arange(h_packed)
        w = torch.arange(w_packed)
        l = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)

        # Pack latents: (B, C, H, W) -> (B, H*W, C)
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        # Prompt embeddings: use random tensors matching joint_attention_dim
        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        # Text IDs (B, seq_len, 4)
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(max_sequence_length)
        text_ids = torch.cartesian_prod(t, h, w, l)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)

        # Guidance
        guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        inputs = {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
