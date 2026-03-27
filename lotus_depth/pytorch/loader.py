# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Lotus Depth model loader implementation for diffusion-based monocular depth estimation.
"""
import torch
from diffusers import DiffusionPipeline
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
    """Available Lotus Depth model variants."""

    DEPTH_D_V1_1 = "Depth-D-v1-1"


class ModelLoader(ForgeModel):
    """Lotus Depth model loader implementation."""

    _VARIANTS = {
        ModelVariant.DEPTH_D_V1_1: ModelConfig(
            pretrained_model_name="jingheya/lotus-depth-d-v1-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEPTH_D_V1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LotusDepth",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        dtype = dtype_override or torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            **kwargs,
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        image = Image.new("RGB", (512, 512))

        inputs = [image] * batch_size
        return inputs
