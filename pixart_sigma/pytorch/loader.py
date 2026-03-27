# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PixArt-Sigma model loader implementation
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
from diffusers import PixArtSigmaPipeline


class ModelVariant(StrEnum):
    """Available PixArt-Sigma model variants."""

    XL_2_1024_MS = "XL-2-1024-MS"


class ModelLoader(ForgeModel):
    """PixArt-Sigma model loader implementation."""

    _VARIANTS = {
        ModelVariant.XL_2_1024_MS: ModelConfig(
            pretrained_model_name="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        )
    }

    DEFAULT_VARIANT = ModelVariant.XL_2_1024_MS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="PixArt-Sigma",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PixArt-Sigma pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            PixArtSigmaPipeline: The pre-trained PixArt-Sigma pipeline.
        """
        dtype = dtype_override or torch.float16
        pipe = PixArtSigmaPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for PixArt-Sigma.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "A small cactus with a happy face in the Sahara desert.",
        ] * batch_size
        return prompt
