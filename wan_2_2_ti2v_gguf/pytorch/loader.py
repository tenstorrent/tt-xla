#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 TI2V 5B GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 Text-and-Image-to-Video transformers from
QuantStack/Wan2.2-TI2V-5B-GGUF and builds a WanPipeline.

The Wan 2.2 TI2V 5B model generates video from text prompts using a
single 5B-parameter transformer. Each variant corresponds to a different
GGUF quantization level.

Available variants:
- WAN22_TI2V_Q4_K_M: Q4_K_M quantization
- WAN22_TI2V_Q8_0: Q8_0 quantization
"""

from typing import Any, Optional

import torch

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

GGUF_REPO = "QuantStack/Wan2.2-TI2V-5B-GGUF"
BASE_PIPELINE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.2 TI2V 5B GGUF variants."""

    WAN22_TI2V_Q4_K_M = "2.2_TI2V_Q4_K_M"
    WAN22_TI2V_Q8_0 = "2.2_TI2V_Q8_0"


_GGUF_FILES = {
    ModelVariant.WAN22_TI2V_Q4_K_M: "Wan2.2-TI2V-5B-Q4_K_M.gguf",
    ModelVariant.WAN22_TI2V_Q8_0: "Wan2.2-TI2V-5B-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.2 TI2V 5B GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_TI2V_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN22_TI2V_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_TI2V_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_TI2V_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the GGUF-quantized Wan 2.2 TI2V transformer and build the pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer,
        then constructs the full WanPipeline with the base model's VAE in float32
        for numerical stability.
        """
        from diffusers import (
            AutoencoderKLWan,
            GGUFQuantizationConfig,
            WanPipeline,
            WanTransformer3DModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        transformer = WanTransformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        vae = AutoencoderKLWan.from_pretrained(
            BASE_PIPELINE,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        self.pipeline = WanPipeline.from_pretrained(
            BASE_PIPELINE,
            transformer=transformer,
            vae=vae,
            torch_dtype=compute_dtype,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-video generation."""
        if prompt is None:
            prompt = (
                "Astronaut in a jungle, cold color palette, muted colors, "
                "detailed, 8k"
            )

        return {
            "prompt": prompt,
            "height": 480,
            "width": 832,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
        }
