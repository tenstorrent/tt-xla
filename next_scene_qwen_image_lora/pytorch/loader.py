#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Next-Scene Qwen Image LoRA model loader implementation.

Loads the Qwen-Image-Edit base pipeline and applies cinematic next-scene LoRA
weights from lovis93/next-scene-qwen-image-lora-2509 for frame-to-frame
image generation with natural visual progression.

Available variants:
- NEXT_SCENE_V2: Recommended v2 LoRA (next-scene_lora-v2-3000.safetensors)
- NEXT_SCENE_V1: Legacy v1 LoRA (next-scene_lora_v1-3000.safetensors)
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

BASE_MODEL = "Qwen/Qwen-Image-2509"
LORA_REPO = "lovis93/next-scene-qwen-image-lora-2509"

# LoRA weight filenames
LORA_V2 = "next-scene_lora-v2-3000.safetensors"
LORA_V1 = "next-scene_lora_v1-3000.safetensors"


class ModelVariant(StrEnum):
    """Available Next-Scene Qwen Image LoRA variants."""

    NEXT_SCENE_V2 = "V2"
    NEXT_SCENE_V1 = "V1"


_LORA_FILES = {
    ModelVariant.NEXT_SCENE_V2: LORA_V2,
    ModelVariant.NEXT_SCENE_V1: LORA_V1,
}


class ModelLoader(ForgeModel):
    """Next-Scene Qwen Image LoRA model loader."""

    _VARIANTS = {
        ModelVariant.NEXT_SCENE_V2: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.NEXT_SCENE_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NEXT_SCENE_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NEXT_SCENE_QWEN_IMAGE_LORA",
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
        """Load the Qwen-Image-Edit pipeline with next-scene LoRA weights applied.

        Returns:
            DiffusionPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for next-scene image generation.

        Returns:
            dict with prompt and image keys.
        """
        if prompt is None:
            prompt = (
                "Next Scene: The camera moves slightly forward as sunlight "
                "breaks through the clouds, casting a soft glow around the "
                "character's silhouette in the mist. Realistic cinematic style, "
                "atmospheric depth."
            )

        # Create a small test image (RGB)
        image = Image.new("RGB", (256, 256), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
