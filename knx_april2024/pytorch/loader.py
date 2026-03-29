# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KNX-APRIl2024 (Ktiseos-Nyx-dev/KNX-APRIl2024) model loader implementation.

KNX-APRIl2024 is a text-to-image model fine-tuned from Stable Diffusion XL
(SDXL) via Pony Diffusion v6.

Available variants:
- KNX_APRIL2024: Ktiseos-Nyx-dev/KNX-APRIl2024 text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline

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


REPO_ID = "Ktiseos-Nyx-dev/KNX-APRIl2024"


class ModelVariant(StrEnum):
    """Available KNX-APRIl2024 model variants."""

    KNX_APRIL2024 = "knx-april2024"


class ModelLoader(ForgeModel):
    """KNX-APRIl2024 model loader implementation."""

    _VARIANTS = {
        ModelVariant.KNX_APRIL2024: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.KNX_APRIL2024

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="knx-april2024",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the KNX-APRIl2024 pipeline.

        Returns:
            StableDiffusionXLPipeline: The KNX-APRIl2024 pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the KNX-APRIl2024 model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
