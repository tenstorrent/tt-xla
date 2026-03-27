# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tongyi-MAI/Z-Image model loader implementation.

Z-Image is an efficient text-to-image generation foundation model based on a
Single-Stream Diffusion Transformer with full Classifier-Free Guidance support.

Available variants:
- Z_IMAGE: Tongyi-MAI/Z-Image text-to-image generation
"""

from typing import Optional

import torch
from diffusers import ZImagePipeline

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


REPO_ID = "Tongyi-MAI/Z-Image"


class ModelVariant(StrEnum):
    """Available Z-Image model variants."""

    Z_IMAGE = "Z-Image"


class ModelLoader(ForgeModel):
    """Z-Image model loader implementation."""

    _VARIANTS = {
        ModelVariant.Z_IMAGE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Z_IMAGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_Image",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Z-Image pipeline.

        Returns:
            ZImagePipeline: The Z-Image pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = ZImagePipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Z-Image model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
