# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DINOv2 model loader implementation for feature extraction (PyTorch/TIMM).
"""

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from datasets import load_dataset
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DINOv2 feature extraction model variants (TIMM)."""

    SMALL_PATCH14_LVD142M = "Small_Patch14_LVD142M"


class ModelLoader(ForgeModel):
    """DINOv2 model loader implementation for feature extraction (PyTorch/TIMM)."""

    _VARIANTS = {
        ModelVariant.SMALL_PATCH14_LVD142M: ModelConfig(
            pretrained_model_name="vit_small_patch14_dinov2.lvd142m",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_PATCH14_LVD142M

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._transform = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DINOv2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DINOv2 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The DINOv2 model instance for feature extraction.
        """
        model_name = self._variant_config.pretrained_model_name

        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self._model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DINOv2 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._transform is None:
            if self._model is None:
                raise RuntimeError("load_model() must be called before load_inputs()")
            data_config = resolve_data_config(self._model.pretrained_cfg)
            self._transform = create_transform(**data_config, is_training=False)

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        pixel_values = self._transform(image).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
