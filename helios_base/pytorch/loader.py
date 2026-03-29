# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helios-Base text-to-video model loader implementation.

Helios-Base is a 14B parameter real-time long video generation model fine-tuned
from Wan2.1-T2V-14B. It uses a pyramid-based autoregressive approach generating
33 frames per chunk with v-prediction and a custom HeliosScheduler.
"""

from typing import Any, Optional

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
    """Available Helios-Base model variants."""

    HELIOS_BASE = "Base"


class ModelLoader(ForgeModel):
    """Helios-Base text-to-video model loader implementation."""

    _VARIANTS = {
        ModelVariant.HELIOS_BASE: ModelConfig(
            pretrained_model_name="BestWishYsh/Helios-Base",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HELIOS_BASE

    DEFAULT_PROMPT = "A cat walks through a sunlit garden, soft lighting, cinematic, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Helios-Base",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self, dtype_override: Optional[torch.dtype] = None
    ) -> DiffusionPipeline:
        pipe_kwargs = {
            "torch_dtype": (
                dtype_override if dtype_override is not None else torch.bfloat16
            ),
        }

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **pipe_kwargs,
        )

        return self.pipeline

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Helios-Base pipeline.

        Args:
            dtype_override: Optional torch dtype to instantiate the pipeline with.

        Returns:
            DiffusionPipeline: The Helios-Base text-to-video pipeline.
        """
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for the Helios-Base pipeline.

        Args:
            prompt: Optional text prompt for video generation.

        Returns:
            dict: Input dictionary with prompt for the pipeline.
        """
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
