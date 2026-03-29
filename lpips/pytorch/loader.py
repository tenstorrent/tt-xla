# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LPIPS (Learned Perceptual Image Patch Similarity) model loader implementation.

LPIPS is a perceptual image quality metric that computes the distance between
two images using deep features from pretrained networks. It takes two images
as input and returns a scalar perceptual distance score.

Reference: https://huggingface.co/zeahub/lpips
"""

import lpips as lpips_lib
from typing import Optional
from PIL import Image
from torchvision import transforms

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


class ModelVariant(StrEnum):
    """Available LPIPS model variants."""

    VGG = "VGG"
    ALEX = "Alex"
    SQUEEZE = "Squeeze"


class ModelLoader(ForgeModel):
    """LPIPS model loader implementation."""

    _VARIANTS = {
        ModelVariant.VGG: ModelConfig(
            pretrained_model_name="zeahub/lpips",
        ),
        ModelVariant.ALEX: ModelConfig(
            pretrained_model_name="zeahub/lpips",
        ),
        ModelVariant.SQUEEZE: ModelConfig(
            pretrained_model_name="zeahub/lpips",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VGG

    # Map variant to lpips net name
    _NET_MAP = {
        ModelVariant.VGG: "vgg",
        ModelVariant.ALEX: "alex",
        ModelVariant.SQUEEZE: "squeeze",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LPIPS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        net = self._NET_MAP[self._variant]
        model = lpips_lib.LPIPS(net=net)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        preprocess = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        img0 = preprocess(Image.new("RGB", (256, 256), color=(255, 0, 0)))
        img1 = preprocess(Image.new("RGB", (256, 256), color=(0, 0, 255)))

        img0 = img0.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        img1 = img1.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            img0 = img0.to(dtype_override)
            img1 = img1.to(dtype_override)

        return (img0, img1)
