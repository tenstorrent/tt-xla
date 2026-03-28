# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
z_image_turbo_sdnq model loader implementation.

Loads the Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32 quantized text-to-image
pipeline. This is a 4-bit SDNQ-quantized variant of Z-Image-Turbo with
SVD rank-32 decomposition, reducing model size by ~71%.

Available variants:
- Z_IMAGE_TURBO_SDNQ: Full quantized Z-Image-Turbo DiT transformer
"""

from typing import Any, Optional

import sdnq  # noqa: F401 -- registers SDNQ quantization support with diffusers
import torch
from diffusers import ZImagePipeline

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

PIPELINE_REPO_ID = "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32"


class ModelVariant(StrEnum):
    """Available z_image_turbo_sdnq model variants."""

    Z_IMAGE_TURBO_SDNQ = "Z-Image-Turbo-SDNQ"


class ModelLoader(ForgeModel):
    """z_image_turbo_sdnq model loader."""

    _VARIANTS = {
        ModelVariant.Z_IMAGE_TURBO_SDNQ: ModelConfig(
            pretrained_model_name=PIPELINE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Z_IMAGE_TURBO_SDNQ

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_IMAGE_TURBO_SDNQ",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the quantized Z-Image-Turbo pipeline."""
        self._pipe = ZImagePipeline.from_pretrained(
            PIPELINE_REPO_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the DiT transformer from the quantized pipeline."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare transformer inputs (latents, timestep, prompt_embeds)."""
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "A photo of an astronaut riding a horse on mars"

        if self._pipe is None:
            self._load_pipeline(dtype)

        prompt_embeds, _ = self._pipe.encode_prompt(
            prompt=prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        num_channels_latents = self._pipe.transformer.in_channels
        vae_scale = self._pipe.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        timestep = torch.tensor([0.5], dtype=dtype)

        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
