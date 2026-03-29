# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Lotus Normal G v1.1 model loader implementation for surface normal estimation.

Lotus is a diffusion-based visual foundation model for dense geometry prediction.
This loader targets the UNet component of the generative normal estimation pipeline.
"""

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
from .src.model_utils import load_unet, prepare_unet_inputs


class ModelVariant(StrEnum):
    """Available Lotus Normal G v1.1 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Lotus Normal G v1.1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="jingheya/lotus-normal-g-v1-1",
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
            model="Lotus Normal G v1.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Lotus Normal UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DConditionModel: The UNet model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        return load_unet(pretrained_model_name, dtype=dtype_override)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Lotus Normal UNet.

        Args:
            dtype_override: Optional torch.dtype to override input dtypes.

        Returns:
            dict: Dictionary of input tensors for the UNet forward pass.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        return prepare_unet_inputs(pretrained_model_name, dtype=dtype_override)
