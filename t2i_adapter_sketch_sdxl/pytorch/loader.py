# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T2I-Adapter Sketch SDXL model loader implementation
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
    load_t2i_adapter_sketch_sdxl_pipe,
    create_sketch_conditioning_image,
    t2i_adapter_sketch_sdxl_preprocessing,
)


class ModelVariant(StrEnum):
    """Available T2I-Adapter Sketch SDXL model variants."""

    T2I_ADAPTER_SKETCH_SDXL_1_0 = "T2I-Adapter_Sketch_SDXL_1.0"


class ModelLoader(ForgeModel):
    """T2I-Adapter Sketch SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.T2I_ADAPTER_SKETCH_SDXL_1_0: ModelConfig(
            pretrained_model_name="TencentARC/t2i-adapter-sketch-sdxl-1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.T2I_ADAPTER_SKETCH_SDXL_1_0

    prompt = "A cat sitting on a windowsill, high quality, detailed"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="T2I-Adapter Sketch SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the T2I-Adapter Sketch SDXL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionXLAdapterPipeline: The pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_t2i_adapter_sketch_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the T2I-Adapter Sketch SDXL model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors for the UNet with T2I-Adapter residuals:
                - latent_model_input (torch.Tensor)
                - timestep (torch.Tensor)
                - prompt_embeds (torch.Tensor)
                - added_cond_kwargs (dict)
                - down_intrablock_additional_residuals (list of torch.Tensor)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        sketch_image = create_sketch_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            down_intrablock_additional_residuals,
        ) = t2i_adapter_sketch_sdxl_preprocessing(
            self.pipeline, self.prompt, sketch_image
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
            down_intrablock_additional_residuals,
        ]
