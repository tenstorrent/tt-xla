# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DDPM (Denoising Diffusion Probabilistic Model) loader implementation
"""

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
import torch
from diffusers import UNet2DModel
from typing import Optional


class ModelVariant(StrEnum):
    """Available DDPM model variants."""

    CELEBAHQ_256 = "google/ddpm-celebahq-256"


class ModelLoader(ForgeModel):
    """DDPM model loader implementation."""

    _VARIANTS = {
        ModelVariant.CELEBAHQ_256: ModelConfig(
            pretrained_model_name="google/ddpm-celebahq-256",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CELEBAHQ_256

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DDPM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DDPM UNet2D model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DModel: The pre-trained unconditional UNet model.
        """
        model = UNet2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            **kwargs,
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DDPM model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample and timestep inputs.
        """
        sample = torch.randn(batch_size, 3, 256, 256)
        timestep = torch.tensor([0])

        if dtype_override is not None:
            sample = sample.to(dtype_override)

        return {
            "sample": sample,
            "timestep": timestep,
        }
