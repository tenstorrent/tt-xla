#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan diffusion model loader implementation.

Supports:
- Full pipeline loading (subfolder=None)
- VAE component loading (subfolder="vae") for encoder/decoder testing

Available variants:
- WAN22_TI2V_5B: Wan 2.2 text-to-image-to-video 5B (full pipeline only)
- WAN22_T2V_A14B: Wan 2.2 text-to-video A14B MoE (full pipeline only)
  Uses Mixture-of-Experts with high/low noise experts (~14B active per step)
- WAN21_T2V_14B: Wan 2.1 text-to-video 14B (supports VAE subfolder)
- WAN21_VACE_1_3B: Wan 2.1 VACE (Video Creation and Editing) 1.3B
  Based on Kijai/WanVideo_comfy, uses Wan-AI/Wan2.1-VACE-1.3B-diffusers
- WAN21_I2V_14B_480P: Wan 2.1 Image-to-Video 14B 480P
  Uses WanImageToVideoPipeline with CLIPVisionModel image encoder
- WAN21_I2V_14B_720P: Wan 2.1 Image-to-Video 14B 720P
  Uses WanImageToVideoPipeline with CLIPVisionModel image encoder (higher res)
"""

from typing import Any, Optional, Dict

import torch
from diffusers import DiffusionPipeline  # type: ignore[import]

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
from .src.utils import (
    load_vae,
    load_vae_decoder_inputs,
    load_vae_encoder_inputs,
    load_vace_pipeline,
    load_vace_inputs,
    load_i2v_pipeline,
    load_i2v_inputs,
)

SUPPORTED_SUBFOLDERS = {"vae"}


class ModelVariant(StrEnum):
    """Available Wan diffusion model variants."""

    WAN22_TI2V_5B = "2.2_Ti2v_5B"
    WAN22_T2V_A14B = "2.2_T2v_A14B"
    WAN21_T2V_14B = "2.1_T2v_14B"
    WAN21_VACE_1_3B = "2.1_VACE_1.3B"
    WAN21_I2V_14B_480P = "2.1_I2v_14B_480P"
    WAN21_I2V_14B_720P = "2.1_I2v_14B_720P"


class ModelLoader(ForgeModel):
    """Wan diffusion model loader that mirrors the standalone inference script."""

    _VARIANTS = {
        ModelVariant.WAN22_TI2V_5B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ),
        ModelVariant.WAN22_T2V_A14B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.2-T2V-A14B",
        ),
        ModelVariant.WAN21_T2V_14B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        ),
        ModelVariant.WAN21_VACE_1_3B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        ),
        ModelVariant.WAN21_I2V_14B_480P: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        ),
        ModelVariant.WAN21_I2V_14B_720P: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_TI2V_5B

    # Reuse the prompt from the reference inference script for smoke testing.
    DEFAULT_PROMPT = (
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, subfolder: Optional[str] = None
    ):
        super().__init__(variant)
        if subfolder is not None and subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        if variant in (
            ModelVariant.WAN22_T2V_A14B,
            ModelVariant.WAN21_VACE_1_3B,
            ModelVariant.WAN21_I2V_14B_480P,
        ):
            group = ModelGroup.VULCAN
            task = ModelTask.MM_VIDEO_TTT
        elif variant == ModelVariant.WAN21_T2V_14B:
            group = ModelGroup.RED
            task = ModelTask.MM_VIDEO_TTT
        else:
            group = ModelGroup.RED
            task = ModelTask.MM_IMAGE_TTT

        return ModelInfo(
            model="WAN",
            variant=variant,
            group=group,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DiffusionPipeline:
        if extra_pipe_kwargs is None:
            extra_pipe_kwargs = {}

        pipe_kwargs = {
            "torch_dtype": (
                dtype_override if dtype_override is not None else torch.float32
            ),
            "device_map": device_map,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        pipe_kwargs.update(extra_pipe_kwargs)

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **pipe_kwargs,
        )

        # Align dtype/device post creation in case caller wants something else
        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Load and return the Wan diffusion pipeline or VAE component.

        Args:
            dtype_override: Optional torch dtype to instantiate/convert the pipeline with.
            device_map: Device placement passed through to DiffusionPipeline.
            low_cpu_mem_usage: Whether to enable the huggingface low-memory loading path.
            extra_pipe_kwargs: Additional kwargs forwarded to DiffusionPipeline.from_pretrained.

        Returns:
            DiffusionPipeline or AutoencoderKLWan depending on subfolder.
        """
        if self._subfolder == "vae":
            dtype = dtype_override if dtype_override is not None else torch.float32
            return load_vae(self._variant_config.pretrained_model_name, dtype)

        if self._variant is not None and self._variant in (
            ModelVariant.WAN21_I2V_14B_480P,
            ModelVariant.WAN21_I2V_14B_720P,
        ):
            dtype = dtype_override if dtype_override is not None else torch.bfloat16
            self.pipeline = load_i2v_pipeline(
                self._variant_config.pretrained_model_name, dtype
            )
            return self.pipeline

        if self._variant is not None and self._variant == ModelVariant.WAN21_VACE_1_3B:
            dtype = dtype_override if dtype_override is not None else torch.float32
            self.pipeline = load_vace_pipeline(
                self._variant_config.pretrained_model_name, dtype
            )
            return self.pipeline

        if self.pipeline is None:
            return self._load_pipeline(
                dtype_override=dtype_override,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                extra_pipe_kwargs=extra_pipe_kwargs,
            )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """
        Prepare inputs for the model or component.

        For VAE subfolder, pass vae_type="decoder" or vae_type="encoder".
        For VACE variant, returns reference-to-video inputs.
        For full pipeline, returns a prompt dict.
        """
        if self._subfolder == "vae":
            dtype = kwargs.get("dtype_override", torch.float32)
            vae_type = kwargs.get("vae_type")
            if vae_type == "decoder":
                return load_vae_decoder_inputs(dtype)
            elif vae_type == "encoder":
                return load_vae_encoder_inputs(dtype)
            else:
                raise ValueError(
                    f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
                )

        if self._variant is not None and self._variant in (
            ModelVariant.WAN21_I2V_14B_480P,
            ModelVariant.WAN21_I2V_14B_720P,
        ):
            height = 720 if self._variant == ModelVariant.WAN21_I2V_14B_720P else 480
            width = 1280 if self._variant == ModelVariant.WAN21_I2V_14B_720P else 832
            return load_i2v_inputs(
                prompt=prompt if prompt is not None else self.DEFAULT_PROMPT,
                height=height,
                width=width,
            )

        if self._variant is not None and self._variant == ModelVariant.WAN21_VACE_1_3B:
            return load_vace_inputs(
                prompt=prompt if prompt is not None else self.DEFAULT_PROMPT
            )

        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
