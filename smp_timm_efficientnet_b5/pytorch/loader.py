# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SMP EfficientNet-B5 model loader implementation (smp-hub/timm-efficientnet-b5.imagenet)
"""

from typing import Optional

import torch
from torchvision import transforms

import segmentation_models_pytorch as smp

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
    """Available SMP EfficientNet-B5 model variants."""

    TIMM_EFFICIENTNET_B5_IMAGENET = "Timm_Efficientnet_B5_Imagenet"


class ModelLoader(ForgeModel):
    """SMP EfficientNet-B5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.TIMM_EFFICIENTNET_B5_IMAGENET: ModelConfig(
            pretrained_model_name="smp-hub/timm-efficientnet-b5.imagenet",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TIMM_EFFICIENTNET_B5_IMAGENET

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SMP-EfficientNet-B5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model = smp.Unet(
            encoder_name="timm-efficientnet-b5",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        model.eval()

        self.model = model

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"].convert("RGB")

        params = smp.encoders.get_preprocessing_params("timm-efficientnet-b5")
        std = torch.tensor(params["std"]).view(1, 3, 1, 1)
        mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)

        img_tensor = transforms.ToTensor()(image).unsqueeze(0)

        # Pad to multiple of 32 for encoder compatibility
        _, _, h, w = img_tensor.shape
        output_stride = 32
        new_h = ((h - 1) // output_stride + 1) * output_stride
        new_w = ((w - 1) // output_stride + 1) * output_stride

        if h != new_h or w != new_w:
            pad_h = new_h - h
            pad_w = new_w - w
            img_tensor = torch.nn.functional.pad(
                img_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        inputs = (img_tensor - mean) / std
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
