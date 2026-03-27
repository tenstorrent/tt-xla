# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Depth Anything 3 (DA3) metric depth estimation model loader.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
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

IMAGENET_DATASET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DATASET_STD = [0.229, 0.224, 0.225]


class DA3Wrapper(nn.Module):
    """Wrapper around DepthAnything3 that takes a preprocessed image tensor
    and returns metric depth prediction."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values)


class ModelVariant(StrEnum):
    """Available Depth Anything 3 model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """Depth Anything 3 metric depth estimation model loader."""

    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="depth-anything/da3metric-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DepthAnything3",
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

        wrapper = DA3Wrapper(model)
        wrapper.eval()

        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)

        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Resize to 518x518 (ViT-L patch size 14, 518/14 = 37 patches)
        rgb = TF.resize(rgb, [518, 518])

        # Apply ImageNet normalization
        rgb = TF.normalize(
            rgb,
            mean=IMAGENET_DATASET_MEAN,
            std=IMAGENET_DATASET_STD,
        )

        rgb = rgb.unsqueeze(0)

        if batch_size > 1:
            rgb = rgb.expand(batch_size, -1, -1, -1)

        if dtype_override is not None:
            rgb = rgb.to(dtype_override)

        return rgb
