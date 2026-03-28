# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MODNet ONNX model loader implementation for portrait matting
"""

import onnx
import torch
from torchvision import transforms
from typing import Optional
from huggingface_hub import hf_hub_download
from datasets import load_dataset

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
    """Available MODNet model variants."""

    MODNET = "MODNet"


class ModelLoader(ForgeModel):
    """MODNet ONNX model loader implementation for portrait matting tasks."""

    _VARIANTS = {
        ModelVariant.MODNET: ModelConfig(
            pretrained_model_name="Xenova/modnet",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MODNET

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transform_image = None
        self.image = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MODNet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _setup_transforms(self):
        image_size = (512, 512)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        return self.transform_image

    def load_model(self, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_path = hf_hub_download(pretrained_model_name, filename="onnx/model.onnx")
        model = onnx.load(model_path)

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
