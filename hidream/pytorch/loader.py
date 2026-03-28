# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HiDream-I1 (HiDream-ai/HiDream-I1-Full) model loader implementation.

HiDream-I1 is a text-to-image generation model based on a Sparse Diffusion
Transformer with 17 billion parameters. It achieves state-of-the-art image
quality and prompt following across photorealistic, cartoon, and artistic styles.

Available variants:
- FULL: HiDream-ai/HiDream-I1-Full text-to-image generation
- DEV: HiDream-ai/HiDream-I1-Dev distilled variant
- FAST: HiDream-ai/HiDream-I1-Fast distilled variant
"""

from typing import Optional

import torch
from diffusers import HiDreamImagePipeline

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


class ModelVariant(StrEnum):
    """Available HiDream-I1 model variants."""

    FULL = "Full"
    DEV = "Dev"
    FAST = "Fast"


class ModelLoader(ForgeModel):
    """HiDream-I1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.FULL: ModelConfig(
            pretrained_model_name="HiDream-ai/HiDream-I1-Full",
        ),
        ModelVariant.DEV: ModelConfig(
            pretrained_model_name="HiDream-ai/HiDream-I1-Dev",
        ),
        ModelVariant.FAST: ModelConfig(
            pretrained_model_name="HiDream-ai/HiDream-I1-Fast",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FULL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HiDream-I1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the HiDream-I1 pipeline.

        Returns:
            HiDreamImagePipeline: The HiDream-I1 pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = HiDreamImagePipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the HiDream-I1 model.

        Returns:
            list: A list of sample text prompts.
        """
        return ["A photo of an astronaut riding a horse on the moon"] * batch_size
