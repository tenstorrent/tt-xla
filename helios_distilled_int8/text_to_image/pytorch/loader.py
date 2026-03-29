# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helios-Distilled-int8 model loader implementation for text-to-image generation.
"""

from typing import Optional, Dict, Any

import torch
from diffusers import DiffusionPipeline

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Helios-Distilled-int8 model variants."""

    HELIOS_DISTILLED_INT8 = "Helios-Distilled-int8"


class ModelLoader(ForgeModel):
    """Helios-Distilled-int8 model loader implementation."""

    _VARIANTS = {
        ModelVariant.HELIOS_DISTILLED_INT8: ModelConfig(
            pretrained_model_name="szwagros/Helios-Distilled-int8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HELIOS_DISTILLED_INT8

    DEFAULT_PROMPT = "A cinematic portrait of a robot in a neon-lit lab"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Helios-Distilled",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
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
        Load and return the Helios-Distilled-int8 text-to-image pipeline.
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
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
