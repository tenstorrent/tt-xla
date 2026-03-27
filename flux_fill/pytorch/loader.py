# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flux Fill model loader implementation for image inpainting
"""
import torch
from diffusers import FluxFillPipeline
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
    """Available Flux Fill model variants."""

    TINY_RANDOM = "Tiny_Random"


class ModelLoader(ForgeModel):
    """Flux Fill model loader implementation for image inpainting tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.TINY_RANDOM: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-flux-fill",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FluxFill",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load pipeline for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            The loaded pipeline instance
        """
        pipe_kwargs = {
            "use_safetensors": True,
        }
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = FluxFillPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Flux Fill transformer model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Flux Fill transformer model instance.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Flux Fill model.

        The Flux Fill transformer expects hidden_states that include the noisy latents
        concatenated with masked image latents and a packed mask along dim=2,
        resulting in in_channels of 384 (64 + 64 + 256).

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        max_sequence_length = 256
        prompt = "A cat sitting on a windowsill"
        height = 128
        width = 128
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Text encoding for CLIP
        text_inputs_clip = self.pipe.tokenizer(
            prompt,
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
            prompt,
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

        # Create latent dimensions
        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))
        seq_len = (height_latent // 2) * (width_latent // 2)

        # Create noisy latents (packed format)
        noise_channels = 64
        latents = torch.randn(
            batch_size * num_images_per_prompt, seq_len, noise_channels, dtype=dtype
        )

        # Create masked image latents (VAE-encoded masked image + packed mask)
        masked_image_channels = 64
        mask_channels = 256
        masked_image_latents = torch.randn(
            batch_size * num_images_per_prompt,
            seq_len,
            masked_image_channels + mask_channels,
            dtype=dtype,
        )

        # Concatenate noisy latents with masked image latents along channel dim
        hidden_states = torch.cat((latents, masked_image_latents), dim=2)

        # Prepare latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": None,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
