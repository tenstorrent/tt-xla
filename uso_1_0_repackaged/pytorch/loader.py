#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
USO 1.0 Repackaged model loader implementation.

Loads the FLUX.1-dev pipeline and applies USO (Unified Style-Subject Optimized)
LoRA weights from Comfy-Org/USO_1.0_Repackaged for style/subject-driven
text-to-image generation.

Available variants:
- USO_1_0_LORA: FLUX.1-dev with USO LoRA applied
"""

from typing import Any, Optional

import numpy as np
import torch
from diffusers import AutoencoderTiny, FluxPipeline  # type: ignore[import]

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

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
LORA_REPO = "Comfy-Org/USO_1.0_Repackaged"
LORA_FILE = "split_files/loras/uso-flux1-dit-lora-v1.safetensors"


class ModelVariant(StrEnum):
    """Available USO 1.0 Repackaged model variants."""

    USO_1_0_LORA = "1.0_LoRA"


class ModelLoader(ForgeModel):
    """USO 1.0 Repackaged model loader using FLUX.1-dev with LoRA."""

    _VARIANTS = {
        ModelVariant.USO_1_0_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.USO_1_0_LORA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="USO_1_0_REPACKAGED",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.float32) -> FluxPipeline:
        """Load FLUX.1-dev pipeline and apply USO LoRA weights."""
        self.pipe = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        # Replace VAE with tiny version for efficiency
        self.pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taef1", torch_dtype=dtype
        )

        # Apply USO LoRA weights
        self.pipe.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_FILE,
        )

        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()
        return self.pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the FLUX transformer with USO LoRA applied.

        Returns:
            FluxTransformer2DModel with USO LoRA weights.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self.pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)
        return self.pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the FLUX transformer.

        Returns:
            dict matching FluxTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        if self.pipe is None:
            self._load_pipeline(dtype)

        max_sequence_length = 256
        prompt = "A stylized portrait in watercolor style"
        num_images_per_prompt = 1
        height = 128
        width = 128
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        # CLIP text encoding
        text_inputs_clip = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.pipe.text_encoder(
            text_inputs_clip.input_ids, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        # T5 text encoding
        text_inputs_t5 = self.pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = self.pipe.text_encoder_2(
            text_inputs_t5.input_ids, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        # Text IDs
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Latents
        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))

        shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
        )

        latents = torch.randn(shape, dtype=dtype)
        latents = latents.view(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent // 2,
            2,
            width_latent // 2,
            2,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size * num_images_per_prompt,
            (height_latent // 2) * (width_latent // 2),
            num_channels_latents * 4,
        )

        # Latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Guidance (FLUX.1-dev uses guidance_scale > 1)
        guidance = torch.full([batch_size], 3.5, dtype=dtype)

        return {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
