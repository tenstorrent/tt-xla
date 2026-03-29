# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLIGEN 1.4 model loader implementation

GLIGEN extends Stable Diffusion with grounded text-to-image generation,
allowing placement of objects at specified bounding box locations.

Reference: https://huggingface.co/masterful/gligen-1-4-generation-text-box
"""

from typing import Optional

import torch
from diffusers import StableDiffusionGLIGENPipeline

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
    """Available GLIGEN model variants."""

    GENERATION_TEXT_BOX = "generation-text-box"


class ModelLoader(ForgeModel):
    """GLIGEN 1.4 model loader implementation."""

    _VARIANTS = {
        ModelVariant.GENERATION_TEXT_BOX: ModelConfig(
            pretrained_model_name="masterful/gligen-1-4-generation-text-box",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERATION_TEXT_BOX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GLIGEN 1.4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GLIGEN pipeline from Hugging Face.

        Returns:
            StableDiffusionGLIGENPipeline: The pre-trained GLIGEN pipeline object.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionGLIGENPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the GLIGEN model.

        Returns:
            dict: A dictionary with prompt, gligen_phrases, gligen_boxes,
                  and gligen_scheduled_sampling_beta.
        """
        prompt = [
            "a waterfall and a modern high speed train in a beautiful forest with fall foliage",
        ] * batch_size
        gligen_phrases = [
            ["a waterfall", "a modern high speed train"],
        ] * batch_size
        gligen_boxes = [
            [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]],
        ] * batch_size
        return {
            "prompt": prompt,
            "gligen_phrases": gligen_phrases,
            "gligen_boxes": gligen_boxes,
            "gligen_scheduled_sampling_beta": 1,
        }
