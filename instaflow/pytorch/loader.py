# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InstaFlow (XCLiu/instaflow_0_9B_from_sd_1_5) model loader implementation.

InstaFlow is a one-step text-to-image model based on Stable Diffusion 1.5,
distilled using Rectified Flow for single-step image generation.

Available variants:
- INSTAFLOW_0_9B: XCLiu/instaflow_0_9B_from_sd_1_5 text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionPipeline

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


REPO_ID = "XCLiu/instaflow_0_9B_from_sd_1_5"


class ModelVariant(StrEnum):
    """Available InstaFlow model variants."""

    INSTAFLOW_0_9B = "InstaFlow_0_9B"


class ModelLoader(ForgeModel):
    """InstaFlow model loader implementation."""

    _VARIANTS = {
        ModelVariant.INSTAFLOW_0_9B: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.INSTAFLOW_0_9B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InstaFlow",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the InstaFlow pipeline.

        Returns:
            StableDiffusionPipeline: The InstaFlow pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the InstaFlow model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "a photo of an astronaut riding a horse on mars",
        ] * batch_size
