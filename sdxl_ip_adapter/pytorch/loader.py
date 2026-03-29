# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXL IP-Adapter model loader implementation
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
    load_ip_adapter_pipe,
    create_ip_adapter_image,
    sdxl_ip_adapter_preprocessing,
)


class ModelVariant(StrEnum):
    """Available SDXL IP-Adapter model variants."""

    SDXL_IP_ADAPTER_VIT_H = "ip-adapter_sdxl_vit-h"


class ModelLoader(ForgeModel):
    """SDXL IP-Adapter model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXL_IP_ADAPTER_VIT_H: ModelConfig(
            pretrained_model_name="OzzyGT/sdxl-ip-adapter",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SDXL_IP_ADAPTER_VIT_H

    prompt = "A beautiful landscape painting in the style of the reference image"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    ip_adapter_subfolder = "."
    ip_adapter_weight_name = "ip-adapter_sdxl_vit-h.safetensors"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXL IP-Adapter",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SDXL pipeline with IP-Adapter.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            DiffusionPipeline: The SDXL pipeline with IP-Adapter loaded.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_ip_adapter_pipe(
            self.base_model,
            pretrained_model_name,
            self.ip_adapter_subfolder,
            self.ip_adapter_weight_name,
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the SDXL IP-Adapter model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors for the UNet with IP-Adapter conditioning:
                - latent_model_input (torch.Tensor)
                - timestep (torch.Tensor)
                - prompt_embeds (torch.Tensor)
                - added_cond_kwargs (dict)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        ip_adapter_image = create_ip_adapter_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
        ) = sdxl_ip_adapter_preprocessing(self.pipeline, self.prompt, ip_adapter_image)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
