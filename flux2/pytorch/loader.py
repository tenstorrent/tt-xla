# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 model loader implementation for text-to-image generation
"""
import torch
import numpy as np
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


class ModelVariant(StrEnum):
    """Available FLUX.2 model variants."""

    DEV = "Dev"


class ModelLoader(ForgeModel):
    """FLUX.2 model loader implementation for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.DEV: ModelConfig(
            pretrained_model_name="black-forest-labs/FLUX.2-dev",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEV

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None
        self.guidance_scale = 4.0

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FLUX.2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        pipe_kwargs = {
            "use_safetensors": True,
        }
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = Flux2Pipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        max_sequence_length = 512
        prompt = "An astronaut riding a horse in a futuristic city"
        num_inference_steps = 1
        height = 128
        width = 128
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        # Encode prompt
        prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        text_ids = text_ids.to(dtype=dtype)

        # Repeat for batch size
        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            text_ids = text_ids.repeat(batch_size, 1, 1)

        # Prepare latents
        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))

        shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents * 4,
            height_latent // 2,
            width_latent // 2,
        )
        latents = torch.randn(shape, dtype=dtype)

        # Prepare latent image IDs (B, H*W, 4)
        latent_ids = self.pipe._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(dtype=dtype)

        # Pack latents: (B, C, H, W) -> (B, H*W, C)
        latents = self.pipe._pack_latents(latents)

        # Prepare guidance
        guidance = torch.full(
            [batch_size * num_images_per_prompt], self.guidance_scale, dtype=dtype
        )

        # Prepare timestep
        timestep = torch.tensor([1.0], dtype=dtype).expand(
            batch_size * num_images_per_prompt
        )

        inputs = {
            "hidden_states": latents,
            "timestep": timestep / 1000,
            "guidance": guidance,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
