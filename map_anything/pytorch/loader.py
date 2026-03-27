# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MapAnything model loader implementation for universal 3D reconstruction.
"""

import torch
import torch.nn as nn
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


class MapAnythingWrapper(nn.Module):
    """Wrapper around MapAnything that takes a preprocessed image tensor
    and returns depth predictions."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        views = [
            {"img": img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)}
            for img in image
        ]
        predictions = self.model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=False,
        )
        return predictions[0]["depth_z"]


class ModelVariant(StrEnum):
    """Available MapAnything model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """MapAnything model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="facebook/map-anything",
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
            model="MapAnything",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from mapanything.models import MapAnything

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = MapAnything.from_pretrained(pretrained_model_name)
        model.eval()

        wrapper = MapAnythingWrapper(model)
        wrapper.eval()

        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)

        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        rgb = rgb.unsqueeze(0)

        if batch_size > 1:
            rgb = rgb.expand(batch_size, -1, -1, -1)

        if dtype_override is not None and rgb.dtype.is_floating_point:
            rgb = rgb.to(dtype_override)

        return rgb
