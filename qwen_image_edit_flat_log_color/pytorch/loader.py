# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit FlatLogColor LoRA model loader implementation.

Loads the Qwen-Image-Edit-2509 base diffusion pipeline and applies the
tlennon-ie/QwenEdit2509-FlatLogColor LoRA weights for converting images
into flat or LOG color profiles suitable for color grading workflows.

Available variants:
- FLAT_LOG_COLOR_2509: FlatLogColor LoRA on Qwen-Image-Edit 2509
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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "tlennon-ie/QwenEdit2509-FlatLogColor"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit FlatLogColor variants."""

    FLAT_LOG_COLOR_2509 = "FlatLogColor_2509"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit FlatLogColor LoRA model loader."""

    _VARIANTS = {
        ModelVariant.FLAT_LOG_COLOR_2509: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FLAT_LOG_COLOR_2509

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_FLAT_LOG_COLOR",
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
        """Load the Qwen-Image-Edit pipeline with FlatLogColor LoRA weights.

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
        """Prepare inputs for flat/LOG color conversion.

        Returns:
            dict with prompt and image keys.
        """
        prompt = "flatcolor"

        # Create a small test image (RGB)
        image = Image.new("RGB", (256, 256), color=(128, 180, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
