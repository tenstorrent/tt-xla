# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UniDepth V2 model loader implementation for monocular metric depth estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from typing import Optional
from dataclasses import dataclass
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

IMAGENET_DATASET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DATASET_STD = [0.229, 0.224, 0.225]


def get_paddings(shape, ratio_bounds):
    """Compute paddings to satisfy aspect ratio constraints."""
    H, W = shape
    ratio = W / H
    if ratio < ratio_bounds[0]:
        new_W = int(H * ratio_bounds[0])
        pad_left = (new_W - W) // 2
        pad_right = new_W - W - pad_left
        pad_top, pad_bottom = 0, 0
    elif ratio > ratio_bounds[1]:
        new_H = int(W / ratio_bounds[1])
        pad_top = (new_H - H) // 2
        pad_bottom = new_H - H - pad_top
        pad_left, pad_right = 0, 0
    else:
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    padded_H = H + pad_top + pad_bottom
    padded_W = W + pad_left + pad_right
    return (pad_left, pad_right, pad_top, pad_bottom), (padded_H, padded_W)


def get_resize_factor(shape, pixels_bounds):
    """Compute resize factor to satisfy pixel count constraints."""
    H, W = shape
    pixels = H * W
    if pixels < pixels_bounds[0]:
        factor = (pixels_bounds[0] / pixels) ** 0.5
    elif pixels > pixels_bounds[1]:
        factor = (pixels_bounds[1] / pixels) ** 0.5
    else:
        factor = 1.0
    new_H = int(H * factor)
    new_W = int(W * factor)
    # Round to multiple of 14 (ViT patch size)
    new_H = max(14, (new_H // 14) * 14)
    new_W = max(14, (new_W // 14) * 14)
    return factor, (new_H, new_W)


class UniDepthV2Wrapper(nn.Module):
    """Wrapper around UniDepthV2 that takes a preprocessed image tensor
    and returns depth prediction."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        inputs = {"image": image, "camera": None}
        _, outputs = self.model.encode_decode(inputs, image_metas=[])
        return outputs["depth"]


@dataclass
class UniDepthV2Config(ModelConfig):
    """Configuration specific to UniDepth V2 models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available UniDepth V2 model variants."""

    VIT_L14 = "ViT-L14"


class ModelLoader(ForgeModel):
    """UniDepth V2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.VIT_L14: UniDepthV2Config(
            pretrained_model_name="lpiccinelli/unidepth-v2-vitl14",
            source=ModelSource.HUGGING_FACE,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_L14

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._shape_constraints = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="UniDepthV2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from unidepth.models import UniDepthV2

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = UniDepthV2.from_pretrained(pretrained_model_name)
        model.eval()

        self._shape_constraints = model.shape_constraints

        wrapper = UniDepthV2Wrapper(model)
        wrapper.eval()

        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)

        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

        _, _, H, W = rgb.unsqueeze(0).shape

        ratio_bounds = self._shape_constraints["ratio_bounds"]
        pixels_bounds = [
            self._shape_constraints["pixels_min"],
            self._shape_constraints["pixels_max"],
        ]

        paddings, (padded_H, padded_W) = get_paddings((H, W), ratio_bounds)
        pad_left, pad_right, pad_top, pad_bottom = paddings
        _, (new_H, new_W) = get_resize_factor((padded_H, padded_W), pixels_bounds)

        rgb = TF.normalize(
            rgb / 255.0,
            mean=IMAGENET_DATASET_MEAN,
            std=IMAGENET_DATASET_STD,
        )
        rgb = F.pad(rgb, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        rgb = F.interpolate(
            rgb.unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False
        )

        if batch_size > 1:
            rgb = rgb.expand(batch_size, -1, -1, -1)

        if dtype_override is not None:
            rgb = rgb.to(dtype_override)

        return rgb
