# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-2512-Turbo-LoRA model loader implementation.

Loads the Qwen-Image-2512 base pipeline and applies the
Wuli-art/Qwen-Image-2512-Turbo-LoRA CFG-distillation LoRA weights
for fast 4-step text-to-image generation.

Available variants:
- QWEN_IMAGE_2512_TURBO_LORA: Turbo LoRA for fast text-to-image generation
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline  # type: ignore[import]

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

BASE_MODEL = "Qwen/Qwen-Image-2512"
LORA_REPO = "Wuli-art/Qwen-Image-2512-Turbo-LoRA"


class ModelVariant(StrEnum):
    """Available Qwen-Image-2512-Turbo-LoRA variants."""

    QWEN_IMAGE_2512_TURBO_LORA = "Image_2512_TurboLoRA"


class ModelLoader(ForgeModel):
    """Qwen-Image-2512-Turbo-LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_2512_TURBO_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_2512_TURBO_LORA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_2512_TURBO_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image-2512 pipeline with Turbo LoRA weights.

        Returns:
            DiffusionPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(LORA_REPO)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-image generation.

        Returns:
            dict with prompt key.
        """
        if prompt is None:
            prompt = (
                "A serene mountain landscape at sunset, "
                "golden light reflecting off a calm lake, detailed and vivid"
            )

        return {
            "prompt": prompt,
        }
