# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Depth Anything V3 (DA3) model loader implementation for monocular depth estimation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from PIL import Image
from datasets import load_dataset

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class DepthAnything3Wrapper(nn.Module):
    """Wrapper around Depth Anything V3 that takes a preprocessed image tensor
    and returns depth prediction."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        prediction = self.model.infer(pixel_values)
        return prediction["depth"]


class ModelVariant(StrEnum):
    """Available Depth Anything V3 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Depth Anything V3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="depth-anything/DA3-BASE",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DepthAnythingV3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from depth_anything_3.api import DepthAnything3

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = DepthAnything3.from_pretrained(pretrained_model_name)
        model.eval()

        wrapper = DepthAnything3Wrapper(model)
        wrapper.eval()

        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)

        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        image_np = np.array(image)
        pixel_values = (
            torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )

        if batch_size > 1:
            pixel_values = pixel_values.expand(batch_size, -1, -1, -1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
