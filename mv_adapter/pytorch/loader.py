# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MV-Adapter T2MV SDXL model loader implementation
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
    load_mv_adapter_pipeline,
    mv_adapter_preprocessing,
)


class ModelVariant(StrEnum):
    """Available MV-Adapter model variants."""

    MV_ADAPTER_T2MV_SDXL = "T2MV_SDXL"


class ModelLoader(ForgeModel):
    """MV-Adapter T2MV SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.MV_ADAPTER_T2MV_SDXL: ModelConfig(
            pretrained_model_name="huanngzh/mv-adapter",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MV_ADAPTER_T2MV_SDXL

    prompt = "a 3D model of a wooden chair, high quality"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MV-Adapter",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MV-Adapter SDXL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            MVAdapterT2MVSDXLPipeline: The pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_mv_adapter_pipeline(pretrained_model_name, self.base_model)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MV-Adapter UNet.

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

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
        ) = mv_adapter_preprocessing(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
        ]
