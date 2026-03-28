# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Marigold Depth model loader implementation for monocular depth estimation.
"""
import torch
from diffusers import MarigoldDepthPipeline
from PIL import Image
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
    """Available Marigold Depth model variants."""

    V1_1 = "v1-1"


class ModelLoader(ForgeModel):
    """Marigold Depth model loader implementation."""

    _VARIANTS = {
        ModelVariant.V1_1: ModelConfig(
            pretrained_model_name="prs-eth/marigold-depth-v1-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MarigoldDepth",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.pipeline = MarigoldDepthPipeline.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        image = Image.new("RGB", (768, 768))

        inputs = [image] * batch_size

        return inputs
