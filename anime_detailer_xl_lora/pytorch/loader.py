# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Anime Detailer XL LoRA model loader implementation
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
from .src.model_utils import load_pipe
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)


class ModelVariant(StrEnum):
    """Available Anime Detailer XL LoRA model variants."""

    ANIME_DETAILER_XL_LORA = "anime-detailer-xl-lora"


class ModelLoader(ForgeModel):
    """Anime Detailer XL LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.ANIME_DETAILER_XL_LORA: ModelConfig(
            pretrained_model_name="Linaqruf/anime-detailer-xl-lora",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANIME_DETAILER_XL_LORA

    # Base SDXL model that the LoRA is applied to
    BASE_MODEL = "Linaqruf/animagine-xl-2.0"
    LORA_FILENAME = "anime-detailer-xl.safetensors"
    LORA_SCALE = 2

    prompt = "face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

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
            model="Anime Detailer XL LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SDXL pipeline with LoRA weights for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionXLPipeline: The pipeline with LoRA weights fused.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(
            base_model_name=self.BASE_MODEL,
            lora_model_id=pretrained_model_name,
            lora_filename=self.LORA_FILENAME,
            lora_scale=self.LORA_SCALE,
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors that can be fed to the model.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
