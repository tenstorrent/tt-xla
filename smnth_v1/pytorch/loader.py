# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Smnth v1 text-to-image LoRA model loader implementation.

This model is a LoRA adapter (ReCodePlus/Smnth_v1_NSFW1) applied on top of
the Tongyi-MAI/Z-Image-Turbo base diffusion model.
"""

import torch
from diffusers import DiffusionPipeline
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
    """Available Smnth v1 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Smnth v1 LoRA text-to-image model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Tongyi-MAI/Z-Image-Turbo",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # LoRA adapter to apply on top of the base model
    _LORA_REPO = "ReCodePlus/Smnth_v1_NSFW1"
    _LORA_WEIGHT_NAME = "Smnth_v1_NSFW1.safetensors"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Smnth v1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the base diffusion pipeline and apply the LoRA adapter.

        Returns:
            DiffusionPipeline: The pipeline with LoRA weights loaded.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        pipe.load_lora_weights(
            self._LORA_REPO,
            weight_name=self._LORA_WEIGHT_NAME,
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample text prompts for the Smnth v1 model.

        The trigger word 'Smnth_v1' must be included in the prompt
        to activate the LoRA's learned concept.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "Smnth_v1 a portrait in a beautiful landscape",
        ] * batch_size
        return prompt
