# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet Depth SD3.5 model loader implementation
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
    load_controlnet_depth_sd3_pipe,
    create_depth_conditioning_image,
    controlnet_depth_sd3_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ControlNet Depth SD3.5 model variants."""

    CONTROLNET_DEPTH_SD3_LARGE = "ControlNet_Depth_SD3.5_Large"


class ModelLoader(ForgeModel):
    """ControlNet Depth SD3.5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_DEPTH_SD3_LARGE: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-large-controlnet-depth",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_DEPTH_SD3_LARGE

    prompt = "a photo of a man"
    base_model = "stabilityai/stable-diffusion-3.5-large"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet Depth SD3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet Depth SD3.5 pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusion3ControlNetPipeline: The pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_depth_sd3_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet Depth SD3.5 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors for the transformer with ControlNet block samples:
                - latent_model_input (torch.Tensor)
                - timestep (torch.Tensor)
                - prompt_embeds (torch.Tensor)
                - pooled_prompt_embeds (torch.Tensor)
                - controlnet_block_samples (tuple of torch.Tensor)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_depth_conditioning_image()

        (
            latent_model_input,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            controlnet_block_samples,
        ) = controlnet_depth_sd3_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype_override)

        return [
            latent_model_input,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            controlnet_block_samples,
        ]
