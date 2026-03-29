# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UnfilteredAI/NSFW-gen-v2 model loader implementation.

NSFW-gen-v2 is a Stable Diffusion XL text-to-image model by UnfilteredAI.

Available variants:
- NSFW_GEN_V2: UnfilteredAI/NSFW-gen-v2 text-to-image generation
"""

from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image

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


REPO_ID = "UnfilteredAI/NSFW-gen-v2"


class ModelVariant(StrEnum):
    """Available NSFW-gen-v2 model variants."""

    NSFW_GEN_V2 = "NSFW_gen_v2"


class ModelLoader(ForgeModel):
    """NSFW-gen-v2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.NSFW_GEN_V2: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NSFW_GEN_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NSFW_gen_v2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NSFW-gen-v2 pipeline.

        Returns:
            AutoPipelineForText2Image: The NSFW-gen-v2 pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the NSFW-gen-v2 model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
