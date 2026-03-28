# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX ControlNet Union model loader implementation
"""

import torch
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
from .src.model_utils import load_flux_controlnet_union_pipe


class ModelVariant(StrEnum):
    """Available FLUX ControlNet Union model variants."""

    FLUX_1_DEV_CONTROLNET_UNION = "FLUX.1-dev-Controlnet-Union"


class ModelLoader(ForgeModel):
    """FLUX ControlNet Union model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX_1_DEV_CONTROLNET_UNION: ModelConfig(
            pretrained_model_name="InstantX/FLUX.1-dev-Controlnet-Union",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLUX_1_DEV_CONTROLNET_UNION

    base_model = "black-forest-labs/FLUX.1-dev"
    prompt = "A bohemian-style female travel blogger with sun-kissed skin and messy beach waves."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX ControlNet Union",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load the FLUX ControlNet Union pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            The loaded pipeline instance
        """
        self.pipe = load_flux_controlnet_union_pipe(
            self._variant_config.pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipe = self.pipe.to(dtype_override)

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX transformer model with ControlNet.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FLUX ControlNet Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        max_sequence_length = 256
        guidance_scale = 3.5
        do_classifier_free_guidance = guidance_scale > 1.0
        height = 128
        width = 128
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        # Text encoding for CLIP
        text_inputs_clip = self.pipe.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_clip = text_inputs_clip.input_ids
        pooled_prompt_embeds = self.pipe.text_encoder(
            text_input_ids_clip, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        # Text encoding for T5
        text_inputs_t5 = self.pipe.tokenizer_2(
            self.prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_t5 = text_inputs_t5.input_ids
        prompt_embeds = self.pipe.text_encoder_2(
            text_input_ids_t5, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        # Create text IDs
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Create latents
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

        # Prepare latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Prepare guidance
        if do_classifier_free_guidance:
            guidance = torch.full([batch_size], guidance_scale, dtype=dtype)
        else:
            guidance = None

        # Prepare inputs
        inputs = {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
