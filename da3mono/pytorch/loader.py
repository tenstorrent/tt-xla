# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Depth Anything 3 Mono model loader implementation for monocular depth estimation.
"""
from PIL import Image
from torchvision import transforms
from depth_anything_3.api import DepthAnything3
from typing import Optional

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
    """Available DA3MONO model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """Depth Anything 3 Mono model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="depth-anything/da3mono-large",
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
            model="DA3Mono",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = DepthAnything3.from_pretrained(pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        image = Image.new("RGB", (518, 518))

        transform = transforms.Compose(
            [
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        inputs = transform(image).unsqueeze(0)

        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
