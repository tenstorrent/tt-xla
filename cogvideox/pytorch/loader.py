# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CogVideoX text-to-video model loader implementation
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
from .src.model_utils import load_pipe, cogvideox_preprocessing


class ModelVariant(StrEnum):
    """Available CogVideoX model variants."""

    COGVIDEOX_5B = "CogVideoX-5b"


class ModelLoader(ForgeModel):
    """CogVideoX text-to-video model loader implementation."""

    _VARIANTS = {
        ModelVariant.COGVIDEOX_5B: ModelConfig(
            pretrained_model_name="zai-org/CogVideoX-5b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COGVIDEOX_5B

    prompt = "A panda playing guitar in a bamboo forest"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CogVideoX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CogVideoX transformer for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The CogVideoX transformer instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the CogVideoX model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors for the transformer:
                - latents (torch.Tensor): Latent video input
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        latents, timestep, prompt_embeds = cogvideox_preprocessing(
            self.pipeline, self.prompt
        )

        if dtype_override:
            latents = latents.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latents, timestep, prompt_embeds]
