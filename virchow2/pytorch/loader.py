# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Virchow2 model loader implementation for histopathology feature extraction.
"""

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from typing import Optional
from datasets import load_dataset

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
    """Available Virchow2 model variants."""

    VIRCHOW2 = "Virchow2"


class ModelLoader(ForgeModel):
    """Virchow2 model loader implementation for histopathology feature extraction."""

    _VARIANTS = {
        ModelVariant.VIRCHOW2: ModelConfig(
            pretrained_model_name="hf-hub:paige-ai/Virchow2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIRCHOW2

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Virchow2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Virchow2 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Virchow2 model instance.
        """
        model_name = self._variant_config.pretrained_model_name

        model = timm.create_model(
            model_name,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self._cached_model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Virchow2 model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        # Load sample image
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        # Use cached model if available, otherwise load it
        if self._cached_model is not None:
            model_for_config = self._cached_model
        else:
            model_for_config = self.load_model(dtype_override=dtype_override)

        # Preprocess image using model's data config
        data_config = resolve_data_config({}, model=model_for_config)
        transforms = create_transform(**data_config)
        inputs = transforms(image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
