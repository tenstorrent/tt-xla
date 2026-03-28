#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 diffusion transformer variants from
befox/WAN2.2-14B-Rapid-AllInOne-GGUF for video generation tasks.

Available variants:
- WAN22_MEGA_V12_Q8: Wan 2.2 Rapid Mega AllInOne v12 (Q8_0 quantization)
"""

from typing import Any, Optional

import torch
from diffusers import WanPipeline, WanTransformer3DModel  # type: ignore[import]
from diffusers.quantizers import GGUFQuantizationConfig  # type: ignore[import]

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

REPO_ID = "befox/WAN2.2-14B-Rapid-AllInOne-GGUF"


class ModelVariant(StrEnum):
    """Available Wan 2.2 GGUF model variants."""

    WAN22_MEGA_V12_Q8 = "2.2_Mega_v12_Q8_0"


# Mapping from variant to GGUF filename within the repo
_GGUF_FILES = {
    ModelVariant.WAN22_MEGA_V12_Q8: "Mega-v12/wan2.2-rapid-mega-aio-v12-Q8_0.gguf",
}

# Base diffusers config repo used to construct the pipeline around the GGUF transformer
_PIPELINE_CONFIG = "Wan-AI/Wan2.2-T2V-14B-Diffusers"


class ModelLoader(ForgeModel):
    """Wan 2.2 GGUF model loader for video generation."""

    _VARIANTS = {
        ModelVariant.WAN22_MEGA_V12_Q8: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_MEGA_V12_Q8

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
            model="WAN_GGUF",
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
        """Load the Wan 2.2 GGUF pipeline.

        Loads the GGUF-quantized transformer and builds a WanPipeline around it.

        Returns:
            WanPipeline instance with GGUF-quantized transformer.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant]

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        transformer = WanTransformer3DModel.from_single_file(
            REPO_ID,
            quantization_config=quantization_config,
            filename=gguf_file,
            torch_dtype=compute_dtype,
        )

        self.pipeline = WanPipeline.from_pretrained(
            _PIPELINE_CONFIG,
            transformer=transformer,
            torch_dtype=compute_dtype,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for the Wan 2.2 GGUF pipeline.

        Returns:
            Dict with prompt for the pipeline.
        """
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
