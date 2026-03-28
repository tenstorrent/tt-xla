# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DDPM CIFAR-10 model loader implementation
"""

import torch
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
from diffusers import DDPMPipeline


class ModelVariant(StrEnum):
    """Available DDPM CIFAR-10 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """DDPM CIFAR-10 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="google/ddpm-cifar10-32",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
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
            model="DDPM CIFAR-10",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DDPM UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            UNet2DModel: The pre-trained UNet model from the DDPM pipeline.
        """
        pipeline = DDPMPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        self.scheduler = pipeline.scheduler

        model = pipeline.unet
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DDPM UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample and timestep inputs.
        """
        dtype = dtype_override or torch.float32

        # Generate random noise as input (32x32 RGB image latents)
        sample = torch.randn((batch_size, 3, 32, 32), dtype=dtype)

        # Use a single timestep
        timestep = torch.tensor([0])

        return {"sample": sample, "timestep": timestep}
