#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Fun Reward LoRA model loader implementation.

Loads the Wan 2.2 T2V base pipeline and applies reward-optimized LoRA weights
from alibaba-pai/Wan2.2-Fun-Reward-LoRAs for text-to-video generation with
improved human preference alignment.

Available variants:
- WAN22_FUN_HIGH_NOISE_HPS21: High noise LoRA optimized with HPS v2.1
- WAN22_FUN_LOW_NOISE_HPS21: Low noise LoRA optimized with HPS v2.1
- WAN22_FUN_HIGH_NOISE_MPS: High noise LoRA optimized with MPS
- WAN22_FUN_LOW_NOISE_MPS: Low noise LoRA optimized with MPS
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

BASE_MODEL = "Wan-AI/Wan2.2-T2V-A14B"
LORA_REPO = "alibaba-pai/Wan2.2-Fun-Reward-LoRAs"

# LoRA weight filenames
LORA_HIGH_NOISE_HPS21 = "Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors"
LORA_LOW_NOISE_HPS21 = "Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors"
LORA_HIGH_NOISE_MPS = "Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors"
LORA_LOW_NOISE_MPS = "Wan2.2-Fun-A14B-InP-low-noise-MPS.safetensors"


class ModelVariant(StrEnum):
    """Available Wan 2.2 Fun Reward LoRA variants."""

    WAN22_FUN_HIGH_NOISE_HPS21 = "2.2_Fun_HighNoise_HPS2.1"
    WAN22_FUN_LOW_NOISE_HPS21 = "2.2_Fun_LowNoise_HPS2.1"
    WAN22_FUN_HIGH_NOISE_MPS = "2.2_Fun_HighNoise_MPS"
    WAN22_FUN_LOW_NOISE_MPS = "2.2_Fun_LowNoise_MPS"


_LORA_FILES = {
    ModelVariant.WAN22_FUN_HIGH_NOISE_HPS21: LORA_HIGH_NOISE_HPS21,
    ModelVariant.WAN22_FUN_LOW_NOISE_HPS21: LORA_LOW_NOISE_HPS21,
    ModelVariant.WAN22_FUN_HIGH_NOISE_MPS: LORA_HIGH_NOISE_MPS,
    ModelVariant.WAN22_FUN_LOW_NOISE_MPS: LORA_LOW_NOISE_MPS,
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Fun Reward LoRA model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_FUN_HIGH_NOISE_HPS21: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.WAN22_FUN_LOW_NOISE_HPS21: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.WAN22_FUN_HIGH_NOISE_MPS: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.WAN22_FUN_LOW_NOISE_MPS: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_FUN_HIGH_NOISE_HPS21

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_FUN_REWARD_LORAS",
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
        """Load the Wan 2.2 T2V pipeline with reward LoRA weights applied.

        Returns:
            DiffusionPipeline with LoRA weights merged.
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
        """Prepare inputs for text-to-video generation.

        Returns:
            dict with prompt key.
        """
        if prompt is None:
            prompt = (
                "A cat walking gracefully across a sunlit garden, "
                "detailed fur texture, cinematic lighting"
            )

        return {
            "prompt": prompt,
        }
