# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Realistic Vision V1.4 Inpainting model loader implementation
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
    load_inpainting_pipe,
    create_dummy_input_image,
    create_dummy_mask_image,
    inpainting_preprocessing,
)


class ModelVariant(StrEnum):
    """Available Realistic Vision V1.4 Inpainting model variants."""

    INPAINTING = "Inpainting"


class ModelLoader(ForgeModel):
    """Realistic Vision V1.4 Inpainting model loader implementation."""

    _VARIANTS = {
        ModelVariant.INPAINTING: ModelConfig(
            pretrained_model_name="Sanster/Realistic_Vision_V1.4-inpainting",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INPAINTING

    prompt = "a clean wall with smooth paint"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Realistic Vision V1.4 Inpainting",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Realistic Vision V1.4 Inpainting pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionInpaintPipeline: The loaded inpainting pipeline.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_inpainting_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the inpainting model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors for the UNet:
                - latent_model_input (torch.Tensor): Concatenated noise + mask + masked image latents
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        input_image = create_dummy_input_image()
        mask_image = create_dummy_mask_image()

        (latent_model_input, timestep, prompt_embeds,) = inpainting_preprocessing(
            self.pipeline, self.prompt, input_image, mask_image
        )

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timestep, prompt_embeds]
