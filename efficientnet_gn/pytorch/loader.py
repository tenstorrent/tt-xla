# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EfficientNet-GN model loader implementation (timm variants)
"""

from typing import Optional

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available EfficientNet-GN model variants (timm)."""

    TEST_EFFICIENTNET_GN_R160_IN1K = "Test_Efficientnet_Gn.r160_in1k"


class ModelLoader(ForgeModel):
    """EfficientNet-GN model loader implementation."""

    _VARIANTS = {
        ModelVariant.TEST_EFFICIENTNET_GN_R160_IN1K: ModelConfig(
            pretrained_model_name="hf_hub:timm/test_efficientnet_gn.r160_in1k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEST_EFFICIENTNET_GN_R160_IN1K

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EfficientNet-GN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._cached_model = None

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        model_for_config = (
            self._cached_model
            if self._cached_model is not None
            else self.load_model(dtype_override=dtype_override)
        )

        data_config = resolve_data_config({}, model=model_for_config)
        data_transforms = create_transform(**data_config)
        inputs = data_transforms(image).unsqueeze(0)
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)
        return inputs
