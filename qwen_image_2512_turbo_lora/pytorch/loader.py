#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Image 2512 Turbo LoRA model loader implementation.

Loads the Qwen Image 2512 base pipeline via DiffSynth-Engine and applies
the 2-step turbo LoRA from Wuli-art/Qwen-Image-2512-Turbo-LoRA-2-Steps
for accelerated text-to-image generation.

Available variants:
- TURBO_2_STEPS: 2-step distilled turbo LoRA for fast inference
"""

import math
from typing import Any, Optional

import torch
from diffsynth_engine import QwenImagePipeline, QwenImagePipelineConfig, fetch_model  # type: ignore[import]

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
LORA_REPO = "Wuli-art/Qwen-Image-2512-Turbo-LoRA-2-Steps"
LORA_WEIGHT_FILE = "Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen Image 2512 Turbo LoRA variants."""

    TURBO_2_STEPS = "Turbo_2Steps"


class ModelLoader(ForgeModel):
    """Qwen Image 2512 Turbo LoRA model loader."""

    _VARIANTS = {
        ModelVariant.TURBO_2_STEPS: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TURBO_2_STEPS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImagePipeline] = None

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
        """Load the Qwen Image 2512 pipeline with turbo LoRA weights applied.

        Returns:
            QwenImagePipeline with LoRA weights fused.
        """
        config = QwenImagePipelineConfig.basic_config(
            model_path=fetch_model(BASE_MODEL, path="transformer/*.safetensors"),
            encoder_path=fetch_model(BASE_MODEL, path="text_encoder/*.safetensors"),
            vae_path=fetch_model(BASE_MODEL, path="vae/*.safetensors"),
            offload_mode="cpu_offload",
        )
        self.pipeline = QwenImagePipeline.from_pretrained(config)

        self.pipeline.load_lora(
            path=fetch_model(LORA_REPO, path=LORA_WEIGHT_FILE),
            scale=1.0,
            fused=True,
        )

        scheduler_config = {
            "exponential_shift_mu": math.log(2.5),
            "use_dynamic_shifting": True,
            "shift_terminal": 0.7155,
        }
        self.pipeline.apply_scheduler_config(scheduler_config)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-image generation.

        Returns:
            dict with prompt and generation parameters.
        """
        if prompt is None:
            prompt = "An astronaut riding a horse in a futuristic city"

        return {
            "prompt": prompt,
            "cfg_scale": 1,
            "num_inference_steps": 2,
            "seed": 42,
            "width": 1328,
            "height": 1328,
        }
