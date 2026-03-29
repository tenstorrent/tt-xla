# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
H0-mini model loader implementation for histology image feature extraction.

H0-mini is a lightweight ViT-B/14 foundation model for histology, distilled from
H-optimus-0 using the DINOv2 self-supervised distillation method.
"""
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from typing import Optional
from PIL import Image

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
    """Available H0-mini model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """H0-mini model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="hf-hub:bioptimus/H0-mini",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._transform = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="H0-mini",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model = timm.create_model(
            model_name,
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model.eval()

        self.model = model
        self._transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self._transform is None and self.model is not None:
            self._transform = create_transform(
                **resolve_data_config(self.model.pretrained_cfg, model=self.model)
            )

        image = Image.new("RGB", (224, 224))

        if self._transform is not None:
            pixel_values = self._transform(image).unsqueeze(0)
        else:
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            pixel_values = transform(image).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
