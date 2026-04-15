# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KNX-APRIl2024 (Ktiseos-Nyx-dev/KNX-APRIl2024) model loader implementation.

KNX-APRIl2024 is a text-to-image model fine-tuned from Stable Diffusion XL
(SDXL) via Pony Diffusion v6.

Available variants:
- KNX_APRIL2024: Ktiseos-Nyx-dev/KNX-APRIl2024 text-to-image generation
"""

from typing import Optional

import torch

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl


REPO_ID = "Ktiseos-Nyx-dev/KNX-APRIl2024"
CHECKPOINT_FILE = "IsekaiFix-April2024.safetensors"


class ModelVariant(StrEnum):
    """Available KNX-APRIl2024 model variants."""

    KNX_APRIL2024 = "knx-april2024"


class ModelLoader(ForgeModel):
    """KNX-APRIl2024 model loader implementation."""

    _VARIANTS = {
        ModelVariant.KNX_APRIL2024: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.KNX_APRIL2024

    prompt = (
        "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="knx-april2024",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the KNX-APRIl2024 UNet.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(pretrained_model_name, CHECKPOINT_FILE)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the KNX-APRIl2024 UNet model.

        Returns:
            dict: Keyword arguments for the UNet forward method.
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

        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
