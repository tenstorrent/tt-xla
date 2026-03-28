# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BiRefNet model loader implementation for dichotomous image segmentation
"""

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from typing import Optional

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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available BiRefNet model variants."""

    BIREFNET = "BiRefNet"
    BIREFNET_PORTRAIT = "BiRefNet-portrait"


class ModelLoader(ForgeModel):
    """BiRefNet model loader implementation for dichotomous image segmentation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BIREFNET: ModelConfig(
            pretrained_model_name="ZhengPeng7/BiRefNet",
        ),
        ModelVariant.BIREFNET_PORTRAIT: ModelConfig(
            pretrained_model_name="ZhengPeng7/BiRefNet-portrait",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BIREFNET

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transform_image = None
        self.image = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BiRefNet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _setup_transforms(self):
        image_size = (1024, 1024)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return self.transform_image

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        model_kwargs["dtype"] = (
            dtype_override if dtype_override is not None else torch.float32
        )
        model_kwargs |= kwargs

        model = AutoModelForImageSegmentation.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )

        torch.set_float32_matmul_precision(["high", "highest"][0])

        if self.transform_image is None:
            self._setup_transforms()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transform_image is None:
            self._setup_transforms()

        dataset = load_dataset("huggingface/cats-image")["test"]
        self.image = dataset[0]["image"]

        inputs = self.transform_image(self.image).unsqueeze(0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        return inputs
