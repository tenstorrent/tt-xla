# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qinglong DetailedEyes Z-Image LoRA model loader implementation.

Loads the Tongyi-MAI/Z-Image-Turbo base pipeline and applies the
bdsqlsz/qinglong_DetailedEyes_Z-Image LoRA adapter for detailed
eye generation in text-to-image diffusion.

Reference: https://huggingface.co/bdsqlsz/qinglong_DetailedEyes_Z-Image
"""

from typing import Optional

import torch
from diffusers import DiffusionPipeline

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
    """Available Qinglong DetailedEyes Z-Image model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Qinglong DetailedEyes Z-Image LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="bdsqlsz/qinglong_DetailedEyes_Z-Image",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    base_model = "Tongyi-MAI/Z-Image-Turbo"
    prompt = "A close-up portrait with highly detailed eyes, photorealistic"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qinglong DetailedEyes Z-Image",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Z-Image-Turbo pipeline with DetailedEyes LoRA.

        Loads the base Tongyi-MAI/Z-Image-Turbo pipeline and applies the
        qinglong_DetailedEyes_Z-Image LoRA adapter weights on top.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            DiffusionPipeline: The pipeline with LoRA weights loaded.
        """
        dtype = dtype_override or torch.float16
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.base_model, torch_dtype=dtype, **kwargs
        )
        self.pipeline.load_lora_weights(
            self._variant_config.pretrained_model_name,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing prompt for text-to-image generation.
        """
        return {
            "prompt": [self.prompt] * batch_size,
        }
