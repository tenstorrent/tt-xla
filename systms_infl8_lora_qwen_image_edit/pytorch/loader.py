#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SYSTMS INFL8 LoRA Qwen Image Edit model loader implementation.

Loads the Qwen-Image-Edit base pipeline and applies the INFL8 LoRA weights
from systms/SYSTMS-INFL8-LoRA-Qwen-Image-Edit-2511 for stylized image editing
that exaggerates or inflates elements within images.

Available variants:
- INFL8_V1: INFL8 LoRA for inflating/exaggerating image elements
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline  # type: ignore[import]
from PIL import Image  # type: ignore[import]

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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "systms/SYSTMS-INFL8-LoRA-Qwen-Image-Edit-2511"

LORA_WEIGHTS = "SYSTMS-INFL8-01C.safetensors"


class ModelVariant(StrEnum):
    """Available SYSTMS INFL8 LoRA variants."""

    INFL8_V1 = "V1"


class ModelLoader(ForgeModel):
    """SYSTMS INFL8 LoRA Qwen Image Edit model loader."""

    _VARIANTS = {
        ModelVariant.INFL8_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.INFL8_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SYSTMS_INFL8_LORA_QWEN_IMAGE_EDIT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image-Edit pipeline with INFL8 LoRA weights applied.

        Returns:
            DiffusionPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHTS,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for INFL8 image editing.

        Returns:
            dict with prompt and image keys.
        """
        if prompt is None:
            prompt = (
                "inflate the balloon, making it grow larger and rounder "
                "with exaggerated proportions and vibrant colors"
            )

        # Create a small test image (RGB)
        image = Image.new("RGB", (256, 256), color=(200, 100, 100))

        return {
            "prompt": prompt,
            "image": image,
        }
