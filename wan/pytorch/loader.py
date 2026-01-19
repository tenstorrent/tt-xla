#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 text-to-image diffusion model loader implementation.
"""

from typing import Optional, Dict, Any

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


class ModelVariant(StrEnum):
    """Available Wan diffusion model variants."""

    WAN22_TI2V_5B = "wan2.2-ti2v-5b"


class ModelLoader(ForgeModel):
    """Wan diffusion model loader that mirrors the standalone inference script."""

    _VARIANTS = {
        ModelVariant.WAN22_TI2V_5B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_TI2V_5B

    # Reuse the prompt from the reference inference script for smoke testing.
    DEFAULT_PROMPT = (
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="wan",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
        device_map: str = "cpu",
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
        dtype_override: Optional[torch.dtype] = None,
        device_map: str = "cpu",
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load and return the Wan diffusion pipeline (DiffusionPipeline).

        Args:
            dtype_override: Optional torch dtype to instantiate/convert the pipeline with.
            device_map: Device placement passed through to DiffusionPipeline.
            low_cpu_mem_usage: Whether to enable the huggingface low-memory loading path.
            extra_pipe_kwargs: Additional kwargs forwarded to DiffusionPipeline.from_pretrained.

        Returns:
            DiffusionPipeline: Ready-to-run Wan text-to-image pipeline.
        """
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

    def load_inputs(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare default text input for the Wan pipeline.

        Args:
            prompt: Optional prompt override; defaults to the reference prompt.

        Returns:
            dict: A dictionary containing the prompt string, matching DiffusionPipeline signature.
        """
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
