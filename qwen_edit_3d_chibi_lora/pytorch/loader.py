#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Edit-3DChibi-LoRA model loader implementation.

Loads the Qwen/Qwen-Image-Edit-2509 base pipeline and applies LoRA weights
from rsshekhawat/Qwen-Edit-3DChibi-LoRA for 3D Chibi style image editing.

Available variants:
- V1: Default LoRA weights (qwen_3d_chibi_lora_v1_000000820.safetensors)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPlusPipeline  # type: ignore[import]
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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "rsshekhawat/Qwen-Edit-3DChibi-LoRA"
LORA_WEIGHT_NAME = "qwen_3d_chibi_lora_v1_000000820.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Edit-3DChibi-LoRA variants."""

    V1 = "v1"


class ModelLoader(ForgeModel):
    """Qwen-Edit-3DChibi-LoRA model loader."""

    _VARIANTS = {
        ModelVariant.V1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_EDIT_3D_CHIBI_LORA",
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
        """Load the Qwen Image Edit pipeline with 3DChibi LoRA weights applied.

        Returns:
            QwenImageEditPlusPipeline with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for 3D Chibi style image editing.

        Returns:
            dict with prompt and image keys.
        """
        if prompt is None:
            prompt = "Convert this image into 3D Chibi Style"

        image = Image.new("RGB", (1024, 1024), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
