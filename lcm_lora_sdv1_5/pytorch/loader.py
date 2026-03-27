# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LCM-LoRA SDv1.5 model loader implementation.

Loads a Stable Diffusion v1.5 base pipeline and applies LCM-LoRA weights
from latent-consistency/lcm-lora-sdv1-5 for fast 2-8 step text-to-image
generation using Latent Consistency Models.
"""

from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler  # type: ignore[import]

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_REPO = "latent-consistency/lcm-lora-sdv1-5"


class ModelVariant(StrEnum):
    """Available LCM-LoRA SDv1.5 variants."""

    LCM_LORA_SDV1_5 = "LCM_LoRA_SDv1.5"


class ModelLoader(ForgeModel):
    """LCM-LoRA SDv1.5 model loader."""

    _VARIANTS = {
        ModelVariant.LCM_LORA_SDV1_5: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LCM_LORA_SDV1_5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[AutoPipelineForText2Image] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LCM_LORA_SDV1_5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the SD v1.5 pipeline with LCM-LoRA weights applied."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        # Swap scheduler for LCM compatibility
        self.pipeline.scheduler = LCMScheduler.from_config(
            self.pipeline.scheduler.config
        )

        # Load and fuse LCM-LoRA weights
        self.pipeline.load_lora_weights(LORA_REPO)
        self.pipeline.fuse_lora()

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs):
        """Prepare inputs for text-to-image generation."""
        if prompt is None:
            prompt = (
                "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
            )

        return {
            "prompt": prompt,
            "num_inference_steps": 4,
            "guidance_scale": 0,
        }
