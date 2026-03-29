# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FastHunyuan video diffusion model loader implementation
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
from .src.model_utils import load_pipe, hunyuan_video_preprocessing


class ModelVariant(StrEnum):
    """Available FastHunyuan video model variants."""

    FAST_HUNYUAN_DIFFUSERS = "FastHunyuan-diffusers"


class ModelLoader(ForgeModel):
    """FastHunyuan video diffusion model loader implementation."""

    _VARIANTS = {
        ModelVariant.FAST_HUNYUAN_DIFFUSERS: ModelConfig(
            pretrained_model_name="FastVideo/FastHunyuan-diffusers",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FAST_HUNYUAN_DIFFUSERS

    prompt = "A cat walks on the grass, realistic style."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FastHunyuan Video",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FastHunyuan video pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            HunyuanVideoPipeline: The loaded pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the FastHunyuan video model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors for the model:
                - latents (torch.Tensor): Latent video input
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
                - prompt_attention_mask (torch.Tensor): Prompt attention mask
                - guidance (torch.Tensor): Guidance scale tensor
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latents,
            timestep,
            prompt_embeds,
            prompt_attention_mask,
            guidance,
        ) = hunyuan_video_preprocessing(self.pipeline, self.prompt)

        if dtype_override:
            latents = latents.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            prompt_attention_mask = prompt_attention_mask.to(dtype_override)
            guidance = guidance.to(dtype_override)

        return [latents, timestep, prompt_embeds, prompt_attention_mask, guidance]
