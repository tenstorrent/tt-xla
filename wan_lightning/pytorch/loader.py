#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 I2V Lightning model loader implementation.

Loads the magespace/Wan2.2-I2V-A14B-Lightning-Diffusers pipeline, a distilled
variant of Wan 2.2 I2V optimized for fast inference with fewer denoising steps.

Available variants:
- WAN22_I2V_A14B_LIGHTNING: Wan 2.2 Image-to-Video A14B Lightning
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


class ModelVariant(StrEnum):
    """Available Wan 2.2 I2V Lightning variants."""

    WAN22_I2V_A14B_LIGHTNING = "2.2_I2V_A14B_Lightning"


class ModelLoader(ForgeModel):
    """Wan 2.2 I2V Lightning model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_I2V_A14B_LIGHTNING: ModelConfig(
            pretrained_model_name="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_I2V_A14B_LIGHTNING

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[WanImageToVideoPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_LIGHTNING",
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
        """Load the Wan 2.2 I2V Lightning pipeline.

        Returns:
            WanImageToVideoPipeline ready for inference.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
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
