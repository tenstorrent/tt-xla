# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EpicRealism XL model loader implementation.

EpicRealism XL v8Kiss is an SDXL-based text-to-image model fine-tuned for
photorealistic image generation with enhanced detail and realism.

Available variants:
- EPICREALISM_XL_V8KISS: misri/epicrealismXL_v8Kiss text-to-image generation
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
from .src.model_utils import load_pipe, epicrealism_xl_preprocessing


class ModelVariant(StrEnum):
    """Available EpicRealism XL model variants."""

    EPICREALISM_XL_V8KISS = "epicrealismXL_v8Kiss"


class ModelLoader(ForgeModel):
    """EpicRealism XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.EPICREALISM_XL_V8KISS: ModelConfig(
            pretrained_model_name="misri/epicrealismXL_v8Kiss",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EPICREALISM_XL_V8KISS

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EpicRealism XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the EpicRealism XL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionXLPipeline: The EpicRealism XL pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the EpicRealism XL model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors that can be fed to the model:
                - latent_model_input (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
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
        ) = epicrealism_xl_preprocessing(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
