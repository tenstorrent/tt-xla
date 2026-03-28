# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Cascade model loader implementation
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
from .src.model_utils import load_prior_pipe, stable_cascade_preprocessing


class ModelVariant(StrEnum):
    """Available Stable Cascade model variants."""

    STABLE_CASCADE_PRIOR = "Prior"
    STABLE_CASCADE_PRIOR_LITE = "Prior_Lite"


class ModelLoader(ForgeModel):
    """Stable Cascade model loader implementation."""

    _VARIANTS = {
        ModelVariant.STABLE_CASCADE_PRIOR: ModelConfig(
            pretrained_model_name="stabilityai/stable-cascade-prior",
        ),
        ModelVariant.STABLE_CASCADE_PRIOR_LITE: ModelConfig(
            pretrained_model_name="stabilityai/stable-cascade-prior",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STABLE_CASCADE_PRIOR

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stable Cascade",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Stable Cascade prior pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableCascadePriorPipeline: The Stable Cascade prior pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_prior_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Stable Cascade prior model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the model:
                - latent_model_input (torch.Tensor): Latent input for the prior
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timestep,
            prompt_embeds,
        ) = stable_cascade_preprocessing(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timestep, prompt_embeds]
