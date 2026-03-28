# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Depth Anything V3 model loader implementation for multi-view depth estimation.
"""
from PIL import Image
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
    """Available Depth Anything V3 model variants."""

    GIANT_1_1 = "GIANT-1.1"


class ModelLoader(ForgeModel):
    """Depth Anything V3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.GIANT_1_1: ModelConfig(
            pretrained_model_name="depth-anything/DA3-GIANT-1.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GIANT_1_1

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
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = DepthAnything3.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        image = Image.new("RGB", (640, 480))
        images = [image] * batch_size

        return (images,)
