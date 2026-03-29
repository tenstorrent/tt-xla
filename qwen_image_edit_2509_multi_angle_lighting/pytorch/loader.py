# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Image Edit 2509 Multi-Angle Lighting LoRA model loader implementation.

Loads the dx8152/Qwen-Edit-2509-Multi-Angle-Lighting LoRA adapter on top of the
Qwen/Qwen-Image-Edit-2509 base diffusion pipeline for multi-angle relighting.
"""

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
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
    """Available Qwen Image Edit Multi-Angle Lighting model variants."""

    V251121 = "v251121"


class ModelLoader(ForgeModel):
    """Qwen Image Edit 2509 Multi-Angle Lighting LoRA model loader."""

    _VARIANTS = {
        ModelVariant.V251121: ModelConfig(
            pretrained_model_name="dx8152/Qwen-Edit-2509-Multi-Angle-Lighting",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V251121

    _BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"

    _LORA_WEIGHT_NAMES = {
        ModelVariant.V251121: "多角度灯光-251121.safetensors",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Qwen Image Edit 2509 Multi-Angle Lighting",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self._BASE_MODEL, torch_dtype=dtype, **kwargs
        )
        pipe.load_lora_weights(
            self._variant_config.pretrained_model_name,
            weight_name=self._LORA_WEIGHT_NAMES[self._variant],
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        luminance_map = Image.new("RGB", (512, 512), color=(255, 255, 255))
        prompt = "使用图2的亮度贴图对图1重新照明(光源来自前方)"
        return {
            "image": [image, luminance_map],
            "prompt": prompt,
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "true_cfg_scale": 4.0,
        }
