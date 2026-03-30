#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AnimateDiff Motion LoRA Pan Right model loader implementation.

Loads the AnimateDiff pipeline with the Stable Diffusion v1.5 base model,
applies the motion adapter (guoyww/animatediff-motion-adapter-v1-5-2),
and loads the pan-right motion LoRA weights from
guoyww/animatediff-motion-lora-pan-right for text-to-video generation
with rightward camera panning.
"""

from typing import Any, Optional

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter  # type: ignore[import]

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

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
LORA_REPO = "guoyww/animatediff-motion-lora-pan-right"


class ModelVariant(StrEnum):
    """Available AnimateDiff Motion LoRA Pan Right variants."""

    PAN_RIGHT = "PanRight"


class ModelLoader(ForgeModel):
    """AnimateDiff Motion LoRA Pan Right model loader."""

    _VARIANTS = {
        ModelVariant.PAN_RIGHT: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.PAN_RIGHT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[AnimateDiffPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ANIMATEDIFF_MOTION_LORA_PAN_RIGHT",
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
        """Load the AnimateDiff pipeline with motion adapter and pan-right LoRA.

        Returns:
            AnimateDiffPipeline with motion adapter and LoRA weights applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        adapter = MotionAdapter.from_pretrained(
            MOTION_ADAPTER,
            torch_dtype=dtype,
        )

        self.pipeline = AnimateDiffPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            motion_adapter=adapter,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(LORA_REPO)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-video generation with pan-right motion.

        Returns:
            dict with prompt key.
        """
        if prompt is None:
            prompt = (
                "A scenic mountain landscape with clouds drifting, "
                "cinematic pan right, smooth camera motion"
            )

        return {
            "prompt": prompt,
        }
