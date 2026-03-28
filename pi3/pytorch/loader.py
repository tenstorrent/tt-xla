# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pi3 model loader implementation for image-to-3D visual geometry reconstruction.

Pi3 (Pi-Cubed) is a permutation-equivariant feed-forward model that predicts
global 3D point clouds, per-view local point maps, confidence scores, and
camera-to-world poses from unordered sets of input images.
"""

import torch
from PIL import Image
from typing import Optional

from ...tools.utils import get_file
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
    """Available Pi3 model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """Pi3 model loader implementation for image-to-3D reconstruction."""

    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="yyfz233/Pi3",
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
            model="Pi3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from pi3.models.pi3 import Pi3

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = Pi3.from_pretrained(pretrained_model_name, **kwargs)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        image_path = get_file(image_url)
        image = Image.open(image_path).convert("RGB")

        # Pi3 expects input as (B, N, 3, H, W) with pixel values in [0, 1]
        # Use 2 views as minimum input for 3D reconstruction
        import torchvision.transforms as T

        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        )

        img_tensor = transform(image)  # (3, H, W)
        # Stack as 2 views: (N=2, 3, H, W)
        imgs = torch.stack([img_tensor, img_tensor], dim=0)
        # Add batch dimension: (B=1, N=2, 3, H, W)
        imgs = imgs.unsqueeze(0)

        if batch_size > 1:
            imgs = imgs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            imgs = imgs.to(dtype_override)

        return imgs
