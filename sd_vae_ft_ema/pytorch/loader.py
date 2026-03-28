# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion VAE (ft-EMA) model loader implementation
"""

import torch
from diffusers import AutoencoderKL
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


class ModelVariant(StrEnum):
    """Available SD-VAE-FT-EMA model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """SD-VAE-FT-EMA model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stabilityai/sd-vae-ft-ema",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SD-VAE-FT-EMA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SD-VAE-FT-EMA model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            AutoencoderKL: The pre-trained VAE model.
        """
        dtype = dtype_override or torch.float32
        model = AutoencoderKL.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SD-VAE-FT-EMA model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            torch.Tensor: Random image tensor suitable for VAE encoding.
        """
        dtype = dtype_override or torch.float32
        # VAE expects 3-channel images at 256x256 resolution
        return torch.randn(batch_size, 3, 256, 256, dtype=dtype)
