# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NSFW-MASTER-Z-IMAGE-TURBO LoRA model loader implementation.

Loads the Tongyi-MAI/Z-Image-Turbo base pipeline and applies the
thutes-gbr25/NSFW-MASTER-Z-IMAGE-TURBO LoRA adapter for text-to-image generation.

Available variants:
- NSFW_MASTER_Z_IMAGE_TURBO: Z-Image-Turbo with NSFW-MASTER LoRA weights applied
"""

from typing import Any, Optional

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

BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
LORA_REPO = "thutes-gbr25/NSFW-MASTER-Z-IMAGE-TURBO"


class ModelVariant(StrEnum):
    """Available NSFW-MASTER-Z-IMAGE-TURBO model variants."""

    NSFW_MASTER_Z_IMAGE_TURBO = "nsfw_master_z_image_turbo"


class ModelLoader(ForgeModel):
    """NSFW-MASTER-Z-IMAGE-TURBO LoRA model loader."""

    _VARIANTS = {
        ModelVariant.NSFW_MASTER_Z_IMAGE_TURBO: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NSFW_MASTER_Z_IMAGE_TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[ZImagePipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NSFW_MASTER_Z_IMAGE_TURBO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load Z-Image-Turbo pipeline with NSFW-MASTER LoRA weights applied.

        Returns:
            The DiT transformer from the pipeline with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.pipeline = ZImagePipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        self.pipeline.load_lora_weights(LORA_REPO)

        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the transformer.

        Returns:
            List of [latent_input_list, timestep, prompt_embeds].
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "A photo of an astronaut riding a horse on mars"

        if self.pipeline is None:
            self.pipeline = ZImagePipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        num_channels_latents = self.pipeline.transformer.in_channels
        vae_scale = self.pipeline.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        timestep = torch.tensor([0.5], dtype=dtype)

        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
