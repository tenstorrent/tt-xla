# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QIE-2511-MP-AnyLight LoRA image-to-image model loader implementation
"""

import torch
from typing import Optional

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
    """Available QIE-2511-MP-AnyLight model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """QIE-2511-MP-AnyLight LoRA image-to-image model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="lilylilith/QIE-2511-MP-AnyLight",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    base_model = "Qwen/Qwen-Image-Edit-2511"
    prompt = "Apply the lighting from image 2 to image 1."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QIE-2511-MP-AnyLight",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the QIE-2511-MP-AnyLight pipeline.

        Loads the base Qwen-Image-Edit-2511 pipeline and applies the
        AnyLight LoRA adapter weights on top.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            DiffusionPipeline: The pipeline with LoRA weights loaded.
        """
        from diffusers import DiffusionPipeline

        dtype = dtype_override or torch.bfloat16
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.base_model, torch_dtype=dtype, **kwargs
        )
        self.pipeline.load_lora_weights(
            self._variant_config.pretrained_model_name,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the QIE-2511-MP-AnyLight model.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing prompt and input images.
        """
        from PIL import Image
        import numpy as np

        # Create a sample source image (to apply lighting to)
        image_1 = Image.fromarray(
            np.random.randint(0, 255, (1080, 1080, 3), dtype=np.uint8)
        )

        # Create a sample reference lighting image
        image_2 = Image.fromarray(
            np.random.randint(0, 255, (1080, 1080, 3), dtype=np.uint8)
        )

        return {
            "prompt": [self.prompt] * batch_size,
            "image": [image_1, image_2],
        }
