# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PhotoMaker V2 model loader implementation
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
    load_photomaker_v2_pipe,
    create_dummy_id_image,
    photomaker_v2_preprocessing,
)


class ModelVariant(StrEnum):
    """Available PhotoMaker V2 model variants."""

    PHOTOMAKER_V2 = "PhotoMaker-V2"


class ModelLoader(ForgeModel):
    """PhotoMaker V2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.PHOTOMAKER_V2: ModelConfig(
            pretrained_model_name="TencentARC/PhotoMaker-V2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PHOTOMAKER_V2

    prompt = "a portrait of a man img wearing sunglasses"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PhotoMaker V2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PhotoMaker V2 pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            PhotoMakerStableDiffusionXLPipeline: The pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_photomaker_v2_pipe(self.base_model, pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the PhotoMaker V2 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors for the UNet:
                - latent_model_input (torch.Tensor)
                - timestep (torch.Tensor)
                - prompt_embeds (torch.Tensor)
                - added_cond_kwargs (dict)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        input_id_images = [create_dummy_id_image()]

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
        ) = photomaker_v2_preprocessing(self.pipeline, self.prompt, input_id_images)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
