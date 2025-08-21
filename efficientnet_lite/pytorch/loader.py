# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EfficientNet-Lite model loader implementation (timm variants)
"""

from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

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
from ...tools.utils import get_file, print_compiled_model_results


class ModelVariant(StrEnum):
    """Available EfficientNet-Lite model variants (timm)."""

    TF_EFFICIENTNET_LITE0_IN1K = "tf_efficientnet_lite0.in1k"
    TF_EFFICIENTNET_LITE1_IN1K = "tf_efficientnet_lite1.in1k"
    TF_EFFICIENTNET_LITE2_IN1K = "tf_efficientnet_lite2.in1k"
    TF_EFFICIENTNET_LITE3_IN1K = "tf_efficientnet_lite3.in1k"
    TF_EFFICIENTNET_LITE4_IN1K = "tf_efficientnet_lite4.in1k"


class ModelLoader(ForgeModel):
    """EfficientNet-Lite model loader implementation."""

    _VARIANTS = {
        ModelVariant.TF_EFFICIENTNET_LITE0_IN1K: ModelConfig(
            pretrained_model_name="tf_efficientnet_lite0.in1k",
        ),
        ModelVariant.TF_EFFICIENTNET_LITE1_IN1K: ModelConfig(
            pretrained_model_name="tf_efficientnet_lite1.in1k",
        ),
        ModelVariant.TF_EFFICIENTNET_LITE2_IN1K: ModelConfig(
            pretrained_model_name="tf_efficientnet_lite2.in1k",
        ),
        ModelVariant.TF_EFFICIENTNET_LITE3_IN1K: ModelConfig(
            pretrained_model_name="tf_efficientnet_lite3.in1k",
        ),
        ModelVariant.TF_EFFICIENTNET_LITE4_IN1K: ModelConfig(
            pretrained_model_name="tf_efficientnet_lite4.in1k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TF_EFFICIENTNET_LITE0_IN1K

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="efficientnet_lite",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._cached_model = None

    def load_model(self, dtype_override=None):
        model_name = self._variant_config.pretrained_model_name
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        # cache for input transform config
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        # Use cached model if available, otherwise load it
        model_for_config = (
            self._cached_model
            if self._cached_model is not None
            else self.load_model(dtype_override)
        )

        data_config = resolve_data_config({}, model=model_for_config)
        data_transforms = create_transform(**data_config)
        inputs = data_transforms(image).unsqueeze(0)
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)
        return inputs

    def print_cls_results(self, compiled_model_out):
        print_compiled_model_results(compiled_model_out)
