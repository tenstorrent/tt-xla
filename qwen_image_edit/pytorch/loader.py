# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit pipeline model loader implementation.

Loads the full Qwen-Image-Edit diffusion pipeline for image editing tasks.
The model takes an input image and a text prompt describing the desired edit,
and produces an edited output image.

Available variants:
- QWEN_IMAGE_EDIT: Qwen-Image-Edit (20B, bf16)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPipeline
from PIL import Image

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

REPO_ID = "Qwen/Qwen-Image-Edit"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit pipeline model variants."""

    QWEN_IMAGE_EDIT = "Qwen_Image_Edit"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit pipeline model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Qwen-Image-Edit pipeline.

        Returns:
            QwenImageEditPipeline instance.
        """
        dtype = dtype_override or torch.bfloat16
        pipeline = QwenImageEditPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return pipeline

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the Qwen-Image-Edit pipeline.

        Returns a dict matching QwenImageEditPipeline.__call__() signature.
        """
        # Create a small sample RGB image for testing
        image = Image.new("RGB", (256, 256), color=(128, 64, 32))

        return {
            "image": image,
            "prompt": "Change the background color to blue.",
            "num_inference_steps": 50,
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "generator": torch.manual_seed(0),
        }
