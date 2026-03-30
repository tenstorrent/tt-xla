# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
4M Tokenizers DINOv2-B14 VQVAE model loader implementation.

Loads the EPFL-VILAB 4M tokenizer that encodes DINOv2-B/14 feature maps
into discrete tokens via a VQVAE with an 8k codebook.
"""

import torch
from fourm.vq.vqvae import VQVAE
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
    """Available 4M Tokenizers DINOv2 model variants."""

    B14_8K_224_448 = "B14_8k_224-448"


class ModelLoader(ForgeModel):
    """4M Tokenizers DINOv2 VQVAE model loader."""

    _VARIANTS = {
        ModelVariant.B14_8K_224_448: ModelConfig(
            pretrained_model_name="EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.B14_8K_224_448

    # DINOv2-B/14 feature map dimensions at 448px input
    _N_CHANNELS = 768
    _FEATURE_SIZE = 28  # 448 / 16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="4M_Tokenizers_DINOv2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the 4M DINOv2 VQVAE tokenizer model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The VQVAE tokenizer model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = VQVAE.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the VQVAE tokenizer.

        The model expects DINOv2-B/14 feature maps of shape [B, 768, 28, 28].

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Random feature map tensor simulating DINOv2-B/14 output.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(
            batch_size,
            self._N_CHANNELS,
            self._FEATURE_SIZE,
            self._FEATURE_SIZE,
            dtype=dtype,
        )
