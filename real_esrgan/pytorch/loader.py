# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Real-ESRGAN model loader implementation for image super-resolution
"""

import torch
from typing import Optional
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

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
from .src.rrdbnet import RRDBNet


class ModelVariant(StrEnum):
    """Available Real-ESRGAN model variants."""

    X4PLUS = "x4plus"


class ModelLoader(ForgeModel):
    """Real-ESRGAN model loader implementation for image super-resolution tasks."""

    _VARIANTS = {
        ModelVariant.X4PLUS: ModelConfig(
            pretrained_model_name="Comfy-Org/Real-ESRGAN_repackaged",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.X4PLUS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Real-ESRGAN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Real-ESRGAN RRDBNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Real-ESRGAN model instance.
        """
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

        # Download and load safetensors weights
        weights_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="RealESRGAN_x4plus.safetensors",
        )
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Real-ESRGAN model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the input tensor.

        Returns:
            torch.Tensor: Input image tensor of shape [batch, 3, 64, 64].
        """
        inputs = torch.randn(batch_size, 3, 64, 64)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
