#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 I2V 14B 480P GGUF model loader implementation.

Loads GGUF-quantized Wan 2.1 Image-to-Video transformer from city96/Wan2.1-I2V-14B-480P-gguf,
combined with standard diffusers pipeline components (VAE, image encoder, text encoder)
from the original Wan-AI/Wan2.1-I2V-14B-480P-Diffusers repo.

Available variants:
- WAN21_I2V_14B_480P_Q4_K_S: Q4_K_S quantization (~10.4 GB)
"""

from typing import Any, Optional

import torch
from diffusers import (
    AutoencoderKLWan,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.quantizers import GGUFQuantizationConfig
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPVisionModel

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

GGUF_REPO_ID = "city96/Wan2.1-I2V-14B-480P-gguf"
DIFFUSERS_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.1 I2V GGUF model variants."""

    WAN21_I2V_14B_480P_Q4_K_S = "I2V_14B_480P_Q4_K_S"


# Map variants to their GGUF filenames
_GGUF_FILES = {
    ModelVariant.WAN21_I2V_14B_480P_Q4_K_S: "wan2.1-i2v-14b-480p-Q4_K_S.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.1 I2V 14B 480P GGUF model loader using diffusers GGUF quantization."""

    _VARIANTS = {
        ModelVariant.WAN21_I2V_14B_480P_Q4_K_S: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_I2V_14B_480P_Q4_K_S

    DEFAULT_PROMPT = (
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_1_I2V_GGUF",
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
        """Load the Wan I2V pipeline with GGUF-quantized transformer.

        The transformer is loaded from the GGUF file, while the VAE and
        image encoder are loaded from the standard diffusers repo in float32
        for numerical stability.

        Returns:
            WanImageToVideoPipeline with GGUF-quantized transformer.
        """
        if self.pipeline is not None:
            if dtype_override is not None:
                self.pipeline = self.pipeline.to(dtype=dtype_override)
            return self.pipeline

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(
            repo_id=GGUF_REPO_ID,
            filename=gguf_filename,
        )

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        transformer = WanTransformer3DModel.from_single_file(
            gguf_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        image_encoder = CLIPVisionModel.from_pretrained(
            DIFFUSERS_REPO_ID,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        )
        vae = AutoencoderKLWan.from_pretrained(
            DIFFUSERS_REPO_ID,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            DIFFUSERS_REPO_ID,
            transformer=transformer,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=compute_dtype,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for the I2V pipeline.

        Returns a dict suitable for passing to WanImageToVideoPipeline.__call__.
        Uses a small synthetic image for testing.
        """
        ref_image = Image.new("RGB", (832, 480), color=(128, 128, 200))

        return {
            "image": ref_image,
            "prompt": prompt if prompt is not None else self.DEFAULT_PROMPT,
            "height": 480,
            "width": 832,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
        }
