#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 SVI v2 PRO LoRA model loader implementation.

Loads the Wan 2.2 I2V base pipeline and applies SVI v2 PRO LoRA weights
from Isi99999/Wan2.2BasedModels for image-to-video generation.

Available variants:
- WAN22_I2V_SVI_HIGH: SVI v2 PRO LoRA (high noise, rank 128)
- WAN22_I2V_SVI_LOW: SVI v2 PRO LoRA (low noise, rank 128)
"""

from typing import Any, Optional

import torch
from diffusers import WanImageToVideoPipeline  # type: ignore[import]
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

BASE_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
LORA_REPO = "Isi99999/Wan2.2BasedModels"

# SVI v2 PRO LoRA weight filenames (rank 128, fp16)
LORA_HIGH = "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors"
LORA_LOW = "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors"


class ModelVariant(StrEnum):
    """Available Wan 2.2 SVI v2 PRO LoRA variants."""

    WAN22_I2V_SVI_HIGH = "2.2_I2V_SVI_High"
    WAN22_I2V_SVI_LOW = "2.2_I2V_SVI_Low"


_LORA_FILES = {
    ModelVariant.WAN22_I2V_SVI_HIGH: LORA_HIGH,
    ModelVariant.WAN22_I2V_SVI_LOW: LORA_LOW,
}


class ModelLoader(ForgeModel):
    """Wan 2.2 SVI v2 PRO LoRA model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_I2V_SVI_HIGH: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.WAN22_I2V_SVI_LOW: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_I2V_SVI_HIGH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[WanImageToVideoPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_SVI_LORAS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Wan 2.2 I2V pipeline with SVI v2 PRO LoRA weights applied.

        Returns:
            WanImageToVideoPipeline with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = WanImageToVideoPipeline.from_pretrained(
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
        """Prepare inputs for image-to-video generation.

        Returns:
            dict with prompt and image keys.
        """
        if prompt is None:
            prompt = (
                "A cat walking gracefully across a sunlit garden, "
                "detailed fur texture, cinematic lighting"
            )

        # Create a small test image (RGB)
        image = Image.new("RGB", (256, 256), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
