#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Remix GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 Remix transformers from
BigDannyPt/Wan-2.2-Remix-GGUF and builds text-to-video or
image-to-video pipelines.

The Wan 2.2 Remix is a community fine-tune of the Wan 2.2 14B model
supporting both text-to-video (T2V) and image-to-video (I2V) generation.
Each mode has high-noise and low-noise expert variants following the
Mixture-of-Experts (MoE) architecture.

Available variants:
- WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: T2V high-noise expert v2.0, Q4_K_M
- WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: I2V high-noise expert v3.0, Q4_K_M
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

GGUF_REPO = "BigDannyPt/Wan-2.2-Remix-GGUF"
T2V_BASE_PIPELINE = "Wan-AI/Wan2.2-T2V-14B-Diffusers"
I2V_BASE_PIPELINE = "Wan-AI/Wan2.2-I2V-14B-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.2 Remix GGUF variants."""

    WAN22_REMIX_T2V_HIGH_V2_Q4_K_M = "2.2_Remix_T2V_High_v2.0_Q4_K_M"
    WAN22_REMIX_I2V_HIGH_V3_Q4_K_M = "2.2_Remix_I2V_High_v3.0_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: "T2V/v2.0/High/wan22RemixT2VI2V_t2vHighV20-Q4_K_M.gguf",
    ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: "I2V/v3.0/High/wan22RemixT2VI2V_i2vHighV30-Q4_K_M.gguf",
}

_IS_I2V = {
    ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: False,
    ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: True,
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Remix GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_REMIX_GGUF",
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
        """Load the GGUF-quantized Wan 2.2 Remix transformer and build the pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer,
        then constructs the appropriate pipeline (T2V or I2V) with the base model's
        VAE in float32 for numerical stability.
        """
        from diffusers import (
            AutoencoderKLWan,
            GGUFQuantizationConfig,
            WanTransformer3DModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        transformer = WanTransformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/resolve/main/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        is_i2v = _IS_I2V[self._variant]
        base_pipeline = I2V_BASE_PIPELINE if is_i2v else T2V_BASE_PIPELINE

        vae = AutoencoderKLWan.from_pretrained(
            base_pipeline,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        if is_i2v:
            from diffusers import WanImageToVideoPipeline
            from transformers import CLIPVisionModel

            image_encoder = CLIPVisionModel.from_pretrained(
                base_pipeline,
                subfolder="image_encoder",
                torch_dtype=torch.float32,
            )

            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                base_pipeline,
                transformer=transformer,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=compute_dtype,
            )
        else:
            from diffusers import WanPipeline

            self.pipeline = WanPipeline.from_pretrained(
                base_pipeline,
                transformer=transformer,
                vae=vae,
                torch_dtype=compute_dtype,
            )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for video generation."""
        if prompt is None:
            prompt = (
                "A cat walking gracefully across a sunlit garden, "
                "detailed fur texture, cinematic lighting"
            )

        is_i2v = _IS_I2V[self._variant]

        if is_i2v:
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

        return {
            "prompt": prompt,
            "height": 480,
            "width": 832,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
        }
