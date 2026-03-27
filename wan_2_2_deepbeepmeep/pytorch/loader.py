#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepBeepMeep/Wan2.2 single-file safetensors model loader implementation.

Loads Wan 2.2 text-to-video diffusion models from single-file safetensors
checkpoints hosted at DeepBeepMeep/Wan2.2.

The Wan 2.2 T2V models use a Mixture-of-Experts (MoE) diffusion transformer
architecture with separate high-noise and low-noise expert checkpoints:
- High-noise expert: handles early denoising steps (overall layout)
- Low-noise expert: handles later denoising steps (detail refinement)

Available variants:
- WAN22_T2V_14B_HIGH_BF16: Text-to-Video 14B high-noise expert, bf16 precision
- WAN22_T2V_5B_BF16: Text-to-Video 5B, bf16 precision
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

SINGLE_FILE_REPO = "DeepBeepMeep/Wan2.2"
BASE_PIPELINE_14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
BASE_PIPELINE_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class ModelVariant(StrEnum):
    """Available DeepBeepMeep/Wan2.2 variants."""

    WAN22_T2V_14B_HIGH_BF16 = "2.2_T2V_14B_HighNoise_bf16"
    WAN22_T2V_5B_BF16 = "2.2_T2V_5B_bf16"


_SINGLE_FILES = {
    ModelVariant.WAN22_T2V_14B_HIGH_BF16: {
        "file": "wan2.2_text2video_14B_high_mbf16.safetensors",
        "base_pipeline": BASE_PIPELINE_14B,
    },
    ModelVariant.WAN22_T2V_5B_BF16: {
        "file": "wan2.2_text2video_5B_mbf16.safetensors",
        "base_pipeline": BASE_PIPELINE_5B,
    },
}


class ModelLoader(ForgeModel):
    """DeepBeepMeep/Wan2.2 single-file safetensors model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_T2V_14B_HIGH_BF16: ModelConfig(
            pretrained_model_name=SINGLE_FILE_REPO,
        ),
        ModelVariant.WAN22_T2V_5B_BF16: ModelConfig(
            pretrained_model_name=SINGLE_FILE_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_T2V_5B_BF16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_DEEPBEEPMEEP",
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
        """Load a single-file Wan 2.2 T2V transformer and build the pipeline.

        Uses diffusers WanTransformer3DModel.from_single_file to load the
        single-file safetensors checkpoint, then constructs the full
        WanPipeline with the base model's scheduler, text encoder, and VAE.
        """
        from diffusers import (
            AutoencoderKLWan,
            WanPipeline,
            WanTransformer3DModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        variant_info = _SINGLE_FILES[self._variant]
        single_file_url = (
            f"https://huggingface.co/{SINGLE_FILE_REPO}"
            f"/resolve/main/{variant_info['file']}"
        )

        transformer = WanTransformer3DModel.from_single_file(
            single_file_url,
            torch_dtype=compute_dtype,
        )

        vae = AutoencoderKLWan.from_pretrained(
            variant_info["base_pipeline"],
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        self.pipeline = WanPipeline.from_pretrained(
            variant_info["base_pipeline"],
            transformer=transformer,
            vae=vae,
            torch_dtype=compute_dtype,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-video generation."""
        if prompt is None:
            prompt = (
                "Astronaut in a jungle, cold color palette, "
                "muted colors, detailed, 8k"
            )

        return {
            "prompt": prompt,
            "height": 480,
            "width": 832,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
        }
