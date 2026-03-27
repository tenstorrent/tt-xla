# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Edit-2509-Multiple-angles LoRA model loader implementation.

Loads the Qwen-Image-Edit-2509 base pipeline and applies the
dx8152/Qwen-Edit-2509-Multiple-angles LoRA weights for camera movement
and angle control in image editing.

Available variants:
- QWEN_EDIT_2509_MULTIPLE_ANGLES: Camera angle control LoRA
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
LORA_REPO = "dx8152/Qwen-Edit-2509-Multiple-angles"


class ModelVariant(StrEnum):
    """Available Qwen-Edit-2509-Multiple-angles variants."""

    QWEN_EDIT_2509_MULTIPLE_ANGLES = "Edit_2509_MultipleAngles"


class ModelLoader(ForgeModel):
    """Qwen-Edit-2509-Multiple-angles LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_EDIT_2509_MULTIPLE_ANGLES: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_EDIT_2509_MULTIPLE_ANGLES

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_EDIT_2509_MULTIPLE_ANGLES",
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
        """Load the Qwen-Image-Edit-2509 pipeline with Multiple-angles LoRA.

        Returns:
            QwenImageEditPlusPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(LORA_REPO)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for image editing with camera angle control.

        Returns:
            dict with prompt and image keys.
        """
        if prompt is None:
            prompt = "Move the camera forward"

        image = Image.new("RGB", (512, 512), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
