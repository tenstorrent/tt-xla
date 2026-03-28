# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-2511-Unblur-Upscale LoRA model loader implementation.

Loads the Qwen-Image-Edit-2511 base pipeline and applies the
prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale LoRA weights for
image unblurring and upscaling.

Available variants:
- QWEN_IMAGE_EDIT_2511_UNBLUR_UPSCALE: Unblur and upscale LoRA
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
LORA_REPO = "prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-2511-Unblur-Upscale variants."""

    QWEN_IMAGE_EDIT_2511_UNBLUR_UPSCALE = "Edit_2511_UnblurUpscale"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-2511-Unblur-Upscale LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_2511_UNBLUR_UPSCALE: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_2511_UNBLUR_UPSCALE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_2511_UNBLUR_UPSCALE",
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
        """Load the Qwen-Image-Edit-2511 pipeline with Unblur-Upscale LoRA.

        Returns:
            DiffusionPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(LORA_REPO)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for image unblurring and upscaling.

        Returns:
            dict with prompt and image keys.
        """
        if prompt is None:
            prompt = "unblur and upscale"

        image = Image.new("RGB", (512, 512), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
