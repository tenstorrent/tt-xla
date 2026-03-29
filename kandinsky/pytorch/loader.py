# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kandinsky 2.1 model loader implementation
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
from diffusers import AutoPipelineForText2Image


class ModelVariant(StrEnum):
    """Available Kandinsky 2.1 model variants."""

    V2_1 = "2.1"


class ModelLoader(ForgeModel):
    """Kandinsky 2.1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.V2_1: ModelConfig(
            pretrained_model_name="kandinsky-community/kandinsky-2-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Kandinsky",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kandinsky 2.1 pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            AutoPipelineForText2Image: The pre-trained Kandinsky 2.1 pipeline.
        """
        dtype = dtype_override or torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for Kandinsky 2.1.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "A alien cheeseburger creature eating itself, claymation",
        ] * batch_size
        return prompt
