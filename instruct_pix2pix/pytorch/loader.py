# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InstructPix2Pix model loader implementation
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
    load_instruct_pix2pix_pipe,
    create_dummy_input_image,
    instruct_pix2pix_preprocessing,
)


class ModelVariant(StrEnum):
    """Available InstructPix2Pix model variants."""

    INSTRUCT_PIX2PIX = "InstructPix2Pix"


class ModelLoader(ForgeModel):
    """InstructPix2Pix model loader implementation."""

    _VARIANTS = {
        ModelVariant.INSTRUCT_PIX2PIX: ModelConfig(
            pretrained_model_name="timbrooks/instruct-pix2pix",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INSTRUCT_PIX2PIX

    prompt = "Turn him into a cyborg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InstructPix2Pix",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the InstructPix2Pix UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model from the InstructPix2Pix pipeline.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_instruct_pix2pix_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the InstructPix2Pix model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors for the UNet:
                - latent_model_input (torch.Tensor): Concatenated noise + image latents
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        input_image = create_dummy_input_image()

        (
            latent_model_input,
            timestep,
            prompt_embeds,
        ) = instruct_pix2pix_preprocessing(self.pipeline, self.prompt, input_image)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timestep, prompt_embeds]
