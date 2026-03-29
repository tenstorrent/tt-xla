# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet Lineart SD1.5 model loader implementation
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
from .src.model_utils import (
    load_controlnet_lineart_sd15_pipe,
    create_lineart_conditioning_image,
    controlnet_lineart_sd15_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ControlNet Lineart SD1.5 model variants."""

    CONTROLNET_LINEART_SD15 = "ControlNet_Lineart_SD1.5"


class ModelLoader(ForgeModel):
    """ControlNet Lineart SD1.5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_LINEART_SD15: ModelConfig(
            pretrained_model_name="ControlNet-1-1-preview/control_v11p_sd15_lineart",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_LINEART_SD15

    prompt = "a detailed line drawing of a cat"
    base_model = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet Lineart SD1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet Lineart SD1.5 pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionControlNetPipeline: The pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_lineart_sd15_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet Lineart SD1.5 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors for the UNet with ControlNet residuals:
                - latent_model_input (torch.Tensor)
                - timestep (torch.Tensor)
                - prompt_embeds (torch.Tensor)
                - down_block_additional_residuals (tuple of torch.Tensor)
                - mid_block_additional_residual (torch.Tensor)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_lineart_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_lineart_sd15_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [
            latent_model_input,
            timesteps,
            prompt_embeds,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ]
