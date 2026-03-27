# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UNI2-h model loader implementation for pathology image feature extraction.
"""

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from datasets import load_dataset
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


class ModelVariant(StrEnum):
    """Available UNI2-h model variants."""

    UNI2_H = "UNI2-h"


class ModelLoader(ForgeModel):
    """UNI2-h model loader for pathology image feature extraction.

    UNI2-h is a ViT-H/14 foundation model for computational pathology with
    681M parameters, pretrained using DINOv2 self-supervised learning on 200M+
    histopathology image tiles from MahmoodLab/UNI2-h.
    """

    _VARIANTS = {
        ModelVariant.UNI2_H: ModelConfig(
            pretrained_model_name="hf-hub:MahmoodLab/UNI2-h",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNI2_H

    # timm model creation kwargs for UNI2-h architecture
    _TIMM_KWARGS = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.transform = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="UNI2-h",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNI2-h model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNI2-h model instance for feature extraction.
        """
        model_name = self._variant_config.pretrained_model_name

        model = timm.create_model(model_name, pretrained=True, **self._TIMM_KWARGS)
        model.eval()

        self.model = model
        self.transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNI2-h model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Preprocessed input tensor of shape [batch_size, 3, 224, 224].
        """
        if self.transform is None:
            if self.model is None:
                self.load_model()
            else:
                self.transform = create_transform(
                    **resolve_data_config(self.model.pretrained_cfg, model=self.model)
                )

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        pixel_values = self.transform(image).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
