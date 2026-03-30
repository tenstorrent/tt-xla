# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T2I-Adapter Sketch SD1.5v2 model loader implementation
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
    load_t2i_adapter_sketch_sd15v2_pipe,
    create_sketch_conditioning_image,
    t2i_adapter_sketch_sd15v2_preprocessing,
)


class ModelVariant(StrEnum):
    """Available T2I-Adapter Sketch SD1.5v2 model variants."""

    T2I_ADAPTER_SKETCH_SD15V2 = "T2I-Adapter_Sketch_SD1.5v2"


class ModelLoader(ForgeModel):
    """T2I-Adapter Sketch SD1.5v2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.T2I_ADAPTER_SKETCH_SD15V2: ModelConfig(
            pretrained_model_name="TencentARC/t2iadapter_sketch_sd15v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.T2I_ADAPTER_SKETCH_SD15V2

    prompt = "royal chamber with fancy bed, high quality, detailed"
    base_model = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="T2I-Adapter Sketch SD1.5v2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the T2I-Adapter Sketch SD1.5v2 pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionAdapterPipeline: The pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_t2i_adapter_sketch_sd15v2_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the T2I-Adapter Sketch SD1.5v2 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors for the UNet with T2I-Adapter residuals:
                - latent_model_input (torch.Tensor)
                - timestep (torch.Tensor)
                - prompt_embeds (torch.Tensor)
                - down_intrablock_additional_residuals (list of torch.Tensor)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        sketch_image = create_sketch_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            down_intrablock_additional_residuals,
        ) = t2i_adapter_sketch_sd15v2_preprocessing(
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
            down_intrablock_additional_residuals,
        ]
