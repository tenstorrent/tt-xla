# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLPMixer lucidrains model loader implementation for image classification
https://github.com/lucidrains/mlp-mixer-pytorch
"""
import torch
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
    """Available MLPMixer lucidrains model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """MLPMixer lucidrains model loader implementation for image classification tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="mlp-mixer",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="mlp-mixer-lucidrains",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the MLPMixer lucidrains model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: MLPMixer lucidrains model instance for image classification.
        """
        from mlp_mixer_pytorch import MLPMixer

        model = MLPMixer(
            image_size=256,
            channels=3,
            patch_size=16,
            dim=512,
            depth=12,
            num_classes=1000,
        )
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the MLPMixer lucidrains model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Input tensor (image data) that can be fed to the model.
        """
        img = torch.randn(batch_size, 3, 256, 256)
        if dtype_override is not None:
            img = img.to(dtype_override)

        return img
