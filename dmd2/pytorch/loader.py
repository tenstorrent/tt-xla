# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DMD2 (Distribution Matching Distillation 2) model loader implementation
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
from .src.model_utils import load_pipe, dmd2_preprocessing


class ModelVariant(StrEnum):
    """Available DMD2 model variants."""

    DMD2_SDXL_4STEP = "SDXL_4step"
    DMD2_SDXL_1STEP = "SDXL_1step"


class ModelLoader(ForgeModel):
    """DMD2 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DMD2_SDXL_4STEP: ModelConfig(
            pretrained_model_name="tianweiy/DMD2",
        ),
        ModelVariant.DMD2_SDXL_1STEP: ModelConfig(
            pretrained_model_name="tianweiy/DMD2",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DMD2_SDXL_4STEP

    # Base SDXL model used by DMD2
    _BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

    # Checkpoint filenames per variant
    _CKPT_NAMES = {
        ModelVariant.DMD2_SDXL_4STEP: "dmd2_sdxl_4step_unet_fp16.bin",
        ModelVariant.DMD2_SDXL_1STEP: "dmd2_sdxl_1step_unet_fp16.bin",
    }

    # Timesteps per variant
    _TIMESTEPS = {
        ModelVariant.DMD2_SDXL_4STEP: [999, 749, 499, 249],
        ModelVariant.DMD2_SDXL_1STEP: [399],
    }

    # Shared configuration parameters
    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DMD2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DMD2 pipeline for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            DiffusionPipeline: The DMD2 pipeline instance with distilled UNet.
        """
        repo_name = self._variant_config.pretrained_model_name
        ckpt_name = self._CKPT_NAMES[self._variant]

        # Load the pipeline with DMD2 distilled UNet
        self.pipeline = load_pipe(self._BASE_MODEL_ID, repo_name, ckpt_name)

        # Apply dtype conversion if specified
        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the DMD2 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the model:
                - latent_model_input (torch.Tensor): Latent input for the UNet
                - timesteps (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
        """
        # Ensure pipeline is initialized
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        timesteps_list = self._TIMESTEPS[self._variant]
        num_inference_steps = len(timesteps_list)

        # Generate preprocessed inputs
        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
        ) = dmd2_preprocessing(
            self.pipeline,
            self.prompt,
            num_inference_steps=num_inference_steps,
            timesteps_list=timesteps_list,
        )

        # Apply dtype conversion if specified
        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
