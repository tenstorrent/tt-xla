# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NoobAI-XL 1.1 (Laxhar/noobai-XL-1.1) model loader implementation.

NoobAI-XL 1.1 is a Stable Diffusion XL based text-to-image generation model.

Available variants:
- NOOBAI_XL_1_1: Laxhar/noobai-XL-1.1 text-to-image generation
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


class ModelVariant(StrEnum):
    """Available NoobAI-XL model variants."""

    NOOBAI_XL_1_1 = "NoobAI_XL_1_1"


class ModelLoader(ForgeModel):
    """NoobAI-XL 1.1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.NOOBAI_XL_1_1: ModelConfig(
            pretrained_model_name="Laxhar/noobai-XL-1.1",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NOOBAI_XL_1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NoobAI_XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NoobAI-XL 1.1 pipeline.

        Returns:
            AutoPipelineForText2Image: The NoobAI-XL 1.1 pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the NoobAI-XL 1.1 model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
