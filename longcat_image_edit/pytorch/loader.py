# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Image-Edit model loader implementation.

Loads the LongCat-Image-Edit-Turbo diffusion transformer for image editing.
Uses LongCatImageEditPipeline from diffusers with a Qwen2.5-VL text encoder.

Available variants:
- TURBO: LongCat-Image-Edit-Turbo (bf16, 8 NFEs)
"""

from typing import Any, Optional

import torch
from diffusers import LongCatImageEditPipeline

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

REPO_ID = "meituan-longcat/LongCat-Image-Edit-Turbo"


class ModelVariant(StrEnum):
    """Available LongCat-Image-Edit model variants."""

    TURBO = "Turbo"


class ModelLoader(ForgeModel):
    """LongCat-Image-Edit model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.TURBO: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LONGCAT_IMAGE_EDIT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> LongCatImageEditPipeline:
        """Load the full LongCat-Image-Edit pipeline."""
        self._pipe = LongCatImageEditPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the LongCat-Image-Edit diffusion transformer.

        Returns:
            LongCatImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the LongCat diffusion transformer.

        Returns a dict matching LongCatImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        if self._pipe is None:
            self._load_pipeline(dtype)

        height = 128
        width = 128
        prompt = "Change the cat to a dog"

        # Encode the prompt using the pipeline's text encoder
        prompt_embeds, pooled_prompt_embeds = self._pipe.encode_prompt(
            prompt=prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        # Prepare latents
        num_channels_latents = self._pipe.transformer.config.in_channels
        vae_scale = self._pipe.vae_scale_factor
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            batch_size, num_channels_latents, latent_h, latent_w, dtype=dtype
        )

        # Prepare timestep
        timestep = torch.tensor([0.5] * batch_size, dtype=dtype)

        # Prepare text IDs and image IDs for positional encoding
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, dtype=dtype)
        img_ids = torch.zeros(latent_h, latent_w, 3, dtype=dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(latent_h)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(latent_w)[None, :]
        img_ids = img_ids.reshape(-1, 3)

        # Pack latents into sequence format: (batch, seq_len, channels)
        latents = latents.reshape(batch_size, num_channels_latents, -1).permute(0, 2, 1)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds.to(dtype),
            "pooled_projections": pooled_prompt_embeds.to(dtype),
            "txt_ids": txt_ids,
            "img_ids": img_ids,
        }
