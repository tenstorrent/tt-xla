# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit Object Adder LoRA model loader implementation.

Loads the Qwen-Image-Edit-2511 base diffusion pipeline and applies the
prithivMLmods/Qwen-Image-Edit-2511-Object-Adder LoRA weights for
object addition in images while preserving background and lighting.

Available variants:
- OBJECT_ADDER_2511: Object Adder LoRA on Qwen-Image-Edit 2511
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline
from PIL import Image

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
LORA_REPO = "prithivMLmods/Qwen-Image-Edit-2511-Object-Adder"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit Object Adder variants."""

    OBJECT_ADDER_2511 = "ObjectAdder_2511"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit Object Adder LoRA model loader."""

    _VARIANTS = {
        ModelVariant.OBJECT_ADDER_2511: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.OBJECT_ADDER_2511

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_OBJECT_ADDER",
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
        """Load the Qwen-Image-Edit pipeline with Object Adder LoRA weights.

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

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for image editing with object addition.

        Returns:
            dict with prompt and image keys.
        """
        prompt = (
            "Add the specified objects to the image while preserving "
            "the background lighting and surrounding elements maintaining "
            "realism and original details."
        )

        # Create a small test image (RGB)
        image = Image.new("RGB", (256, 256), color=(128, 180, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
