# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Depth Anything V3 model loader implementation for monocular depth estimation.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
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
    """Wrapper around DepthAnything3 that takes a preprocessed image tensor
    and returns depth prediction."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values)


class ModelVariant(StrEnum):
    """Available Depth Anything V3 model variants."""

    LARGE_1_1 = "Large-1.1"


class ModelLoader(ForgeModel):
    """Depth Anything V3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE_1_1: ModelConfig(
            pretrained_model_name="depth-anything/DA3-LARGE-1.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_1_1

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

        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        rgb = rgb.unsqueeze(0)

        if batch_size > 1:
            rgb = rgb.expand(batch_size, -1, -1, -1)

        if dtype_override is not None:
            rgb = rgb.to(dtype_override)

        return rgb
