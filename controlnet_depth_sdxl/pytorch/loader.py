# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet Depth SDXL model loader implementation
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
    load_controlnet_depth_sdxl_pipe,
    create_depth_conditioning_image,
    controlnet_depth_sdxl_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ControlNet Depth SDXL model variants."""

    CONTROLNET_DEPTH_SDXL_1_0 = "Depth_SDXL_1.0"


class ModelLoader(ForgeModel):
    """ControlNet Depth SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_DEPTH_SDXL_1_0: ModelConfig(
            pretrained_model_name="xinsir/controlnet-depth-sdxl-1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_DEPTH_SDXL_1_0

    prompt = "A scenic landscape with mountains and a river, high quality"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet Depth SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet Depth SDXL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionXLControlNetPipeline: The pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_depth_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet Depth SDXL model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors for the UNet with ControlNet residuals:
                - latent_model_input (torch.Tensor)
                - timestep (torch.Tensor)
                - prompt_embeds (torch.Tensor)
                - added_cond_kwargs (dict)
                - down_block_additional_residuals (tuple of torch.Tensor)
                - mid_block_additional_residual (torch.Tensor)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_depth_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_depth_sdxl_preprocessing(
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
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ]
