#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 VACE 14B GGUF model loader implementation.

Loads GGUF-quantized Wan 2.1 VACE transformers from
QuantStack/Wan2.1_14B_VACE-GGUF and builds a WanVACEPipeline.

The Wan 2.1 VACE (Video All-in-one Creation Engine) model supports
versatile video creation and editing tasks including reference-to-video
generation. This loader uses GGUF-quantized weights for reduced memory
usage.

Available variants:
- WAN21_VACE_Q4_K_M: Q4_K_M quantization
- WAN21_VACE_Q8_0: Q8_0 quantization
"""

from typing import Any, Optional

import torch
from PIL import Image

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

GGUF_REPO = "QuantStack/Wan2.1_14B_VACE-GGUF"
BASE_PIPELINE = "Wan-AI/Wan2.1-VACE-14B-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.1 VACE 14B GGUF variants."""

    WAN21_VACE_Q4_K_M = "2.1_VACE_Q4_K_M"
    WAN21_VACE_Q8_0 = "2.1_VACE_Q8_0"


_GGUF_FILES = {
    ModelVariant.WAN21_VACE_Q4_K_M: "Wan2.1_14B_VACE-Q4_K_M.gguf",
    ModelVariant.WAN21_VACE_Q8_0: "Wan2.1_14B_VACE-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.1 VACE 14B GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN21_VACE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN21_VACE_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_VACE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_1_VACE_GGUF",
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
        """Load the GGUF-quantized Wan 2.1 VACE transformer and build the pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer,
        then constructs the full WanVACEPipeline with the base model's VAE in
        float32 for numerical stability.
        """
        from diffusers import (
            AutoencoderKLWan,
            GGUFQuantizationConfig,
            WanTransformer3DModel,
            WanVACEPipeline,
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

        self.pipeline = WanVACEPipeline.from_pretrained(
            BASE_PIPELINE,
            transformer=transformer,
            vae=vae,
            torch_dtype=compute_dtype,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for VACE reference-to-video generation."""
        if prompt is None:
            prompt = (
                "A character walking gracefully across a sunlit garden, "
                "smooth animation, detailed motion, cinematic lighting"
            )

        ref_image = Image.new("RGB", (832, 480), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "reference_images": [ref_image],
            "height": 480,
            "width": 832,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
        }
