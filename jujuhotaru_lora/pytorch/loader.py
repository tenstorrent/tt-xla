# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
JujoHotaru LoRA Stable Diffusion model loader implementation
"""

import torch
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
from diffusers import StableDiffusionPipeline
from typing import Optional


class ModelVariant(StrEnum):
    """Available JujoHotaru LoRA model variants."""

    EYECOLLE_ACHILLEA = "eyecolle_achillea"


class ModelLoader(ForgeModel):
    """JujoHotaru LoRA Stable Diffusion model loader implementation."""

    _VARIANTS = {
        ModelVariant.EYECOLLE_ACHILLEA: ModelConfig(
            pretrained_model_name="JujoHotaru/lora",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EYECOLLE_ACHILLEA

    _LORA_WEIGHT_NAMES = {
        ModelVariant.EYECOLLE_ACHILLEA: "eyecolle/eyecolle_achillea_v100.safetensors",
    }

    _BASE_MODEL = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="JujoHotaru LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self._BASE_MODEL, torch_dtype=dtype, **kwargs
        )
        pipe.load_lora_weights(
            self._variant_config.pretrained_model_name,
            weight_name=self._LORA_WEIGHT_NAMES[self._variant],
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        prompt = [
            "1girl, achillea eyes, looking at viewer, upper body",
        ] * batch_size
        return prompt
