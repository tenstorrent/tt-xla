# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-schnell GGUF model loader implementation for text-to-image generation.

This loader uses GGUF-quantized variants of the FLUX.1-schnell model from
lllyasviel/FLUX.1-schnell-gguf. The GGUF transformer is loaded via diffusers'
FluxTransformer2DModel.from_single_file and plugged into a FluxPipeline built
from the original black-forest-labs/FLUX.1-schnell repository.

Available variants:
- Q4_0: 4-bit quantization (default)
- Q8_0: 8-bit quantization
"""

from typing import Optional

import torch
from diffusers import AutoencoderTiny, FluxPipeline, FluxTransformer2DModel

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

GGUF_REPO = "lllyasviel/FLUX.1-schnell-gguf"
BASE_REPO = "black-forest-labs/FLUX.1-schnell"


class ModelVariant(StrEnum):
    """Available FLUX.1-schnell GGUF quantization variants."""

    Q4_0 = "Q4_0"
    Q8_0 = "Q8_0"


# Map variants to their GGUF filenames
_GGUF_FILES = {
    ModelVariant.Q4_0: "flux1-schnell-Q4_0.gguf",
    ModelVariant.Q8_0: "flux1-schnell-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """FLUX.1-schnell GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.Q4_0: ModelConfig(pretrained_model_name=GGUF_REPO),
        ModelVariant.Q8_0: ModelConfig(pretrained_model_name=GGUF_REPO),
    }

    DEFAULT_VARIANT = ModelVariant.Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-schnell GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16):
        """Load the FluxPipeline with a GGUF-quantized transformer."""
        gguf_file = _GGUF_FILES[self._variant]

        transformer = FluxTransformer2DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
            torch_dtype=dtype,
        )

        self.pipe = FluxPipeline.from_pretrained(
            BASE_REPO,
            transformer=transformer,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        self.pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taef1", torch_dtype=dtype
        )

        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX transformer.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipe is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype=dtype_override)
        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the FLUX transformer.

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipe is None:
            self._load_pipeline(dtype)

        max_sequence_length = 256
        prompt = "An astronaut riding a horse in a futuristic city"
        height = 128
        width = 128
        num_images_per_prompt = 1
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

        return {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": None,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
