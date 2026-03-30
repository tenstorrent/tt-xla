# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DASIWA model loader implementation for text-to-image generation.

DASIWA is a LoRA adapter on zai-org/GLM-Image for text-to-image generation.
"""

from typing import Optional

import torch
from diffusers import DiffusionPipeline

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DASIWA model variants."""

    DASIWA = "DASIWA"


class ModelLoader(ForgeModel):
    """DASIWA model loader implementation."""

    _VARIANTS = {
        ModelVariant.DASIWA: ModelConfig(
            pretrained_model_name="thatboymentor/DASIWA",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DASIWA

    DEFAULT_PROMPT = "A beautiful landscape painting in vivid colors"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DASIWA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DASIWA text-to-image pipeline.

        Loads the base zai-org/GLM-Image model and applies the DASIWA LoRA weights.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            DiffusionPipeline: The pipeline with LoRA weights applied.
        """
        dtype = dtype_override or torch.bfloat16
        base_model = "zai-org/GLM-Image"

        self.pipeline = DiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            **kwargs,
        )

        self.pipeline.load_lora_weights(
            self._variant_config.pretrained_model_name,
        )

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for DASIWA.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            dict: A dictionary containing the prompt for image generation.
        """
        prompt = [self.DEFAULT_PROMPT] * batch_size
        return {"prompt": prompt}
