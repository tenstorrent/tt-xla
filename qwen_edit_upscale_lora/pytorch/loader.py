# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Edit 2509 Upscale LoRA model loader implementation.

Loads the Qwen/Qwen-Image-Edit-2509 base pipeline and applies upscale
enhancement LoRA weights from vafipas663/Qwen-Edit-2509-Upscale-LoRA for
image upscaling and enhancement (up to 16x).

Two LoRA checkpoints are chained sequentially:
- LoRA A: qwen-edit-enhance_64-v3_000001000 (rank-64 base enhancement)
- LoRA B: qwen-edit-enhance_000004250 (refinement)

Available variants:
- UPSCALE_LORA: Full two-stage LoRA enhancement pipeline
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
LORA_REPO = "vafipas663/Qwen-Edit-2509-Upscale-LoRA"

# LoRA weight filenames (applied sequentially)
LORA_A = "qwen-edit-enhance_64-v3_000001000.safetensors"
LORA_B = "qwen-edit-enhance_000004250.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Edit 2509 Upscale LoRA variants."""

    UPSCALE_LORA = "Upscale_LoRA"


class ModelLoader(ForgeModel):
    """Qwen-Edit 2509 Upscale LoRA model loader."""

    _VARIANTS = {
        ModelVariant.UPSCALE_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.UPSCALE_LORA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_EDIT_UPSCALE_LORA",
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
        """Load the Qwen-Image-Edit pipeline with upscale LoRA weights applied.

        Returns:
            QwenImageEditPlusPipeline with both LoRA weight stages merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        # Chain LoRA A then LoRA B sequentially
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_A,
            adapter_name="enhance_base",
        )
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_B,
            adapter_name="enhance_refine",
        )
        self.pipeline.set_adapters(["enhance_base", "enhance_refine"])

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for image upscale/enhancement.

        Returns:
            dict with prompt and image keys.
        """
        if prompt is None:
            prompt = (
                "Enhance image quality, a sunlit landscape with rolling green hills "
                "and a clear blue sky"
            )

        # Create a small test image simulating a low-resolution input
        image = Image.new("RGB", (256, 256), color=(100, 150, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
