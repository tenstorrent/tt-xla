#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Animate 14B GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 Animate transformers from
QuantStack/Wan2.2-Animate-14B-GGUF and builds a WanImageToVideoPipeline.

The Wan 2.2 Animate model is a character animation model built on
Wan2.2-I2V-A14B that can animate characters to mimic human motion
from a source video. It uses a Mixture-of-Experts (MoE) architecture
with ~14B active parameters per inference step.

Available variants:
- WAN22_ANIMATE_Q4_K_M: Q4_K_M quantization (~10.7 GB)
- WAN22_ANIMATE_Q8_0: Q8_0 quantization (~17.4 GB)
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

GGUF_REPO = "QuantStack/Wan2.2-Animate-14B-GGUF"
BASE_PIPELINE = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.2 Animate 14B GGUF variants."""

    WAN22_ANIMATE_Q4_K_M = "2.2_Animate_Q4_K_M"
    WAN22_ANIMATE_Q8_0 = "2.2_Animate_Q8_0"


_GGUF_FILES = {
    ModelVariant.WAN22_ANIMATE_Q4_K_M: "Wan2.2-Animate-14B-Q4_K_M.gguf",
    ModelVariant.WAN22_ANIMATE_Q8_0: "Wan2.2-Animate-14B-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Animate 14B GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_ANIMATE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN22_ANIMATE_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_ANIMATE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_ANIMATE_GGUF",
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
        """Load the GGUF-quantized Wan 2.2 Animate transformer and build the pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer,
        then constructs the full WanImageToVideoPipeline with the base model's
        VAE and image encoder in float32 for numerical stability.
        """
        from diffusers import (
            AutoencoderKLWan,
            GGUFQuantizationConfig,
            WanImageToVideoPipeline,
            WanTransformer3DModel,
        )
        from transformers import CLIPVisionModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        transformer = WanTransformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        image_encoder = CLIPVisionModel.from_pretrained(
            BASE_PIPELINE,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        )
        vae = AutoencoderKLWan.from_pretrained(
            BASE_PIPELINE,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            BASE_PIPELINE,
            transformer=transformer,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=compute_dtype,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for character animation generation."""
        if prompt is None:
            prompt = (
                "A character walking gracefully across a sunlit garden, "
                "smooth animation, detailed motion, cinematic lighting"
            )

        image = Image.new("RGB", (832, 480), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "image": image,
            "height": 480,
            "width": 832,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
        }
