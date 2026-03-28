#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio Open 1.0 ComfyUI Repackaged model loader implementation.

Loads single-file safetensors from Comfy-Org/stable-audio-open-1.0_repackaged.
Supports loading the StableAudioDiTModel transformer component for testing.

Available variants:
- STABLE_AUDIO_OPEN_1_0: Stable Audio Open 1.0 DiT transformer
"""

from typing import Any, Optional

import torch
from diffusers import StableAudioPipeline  # type: ignore[import]
from huggingface_hub import hf_hub_download  # type: ignore[import]

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

REPO_ID = "Comfy-Org/stable-audio-open-1.0_repackaged"
ORIGINAL_CONFIG_REPO = "stabilityai/stable-audio-open-1.0"

# DiT model hidden size (from Stable Audio Open 1.0 config)
HIDDEN_SIZE = 1024


class ModelVariant(StrEnum):
    """Available Stable Audio Open ComfyUI Repackaged model variants."""

    STABLE_AUDIO_OPEN_1_0 = "1.0"


class ModelLoader(ForgeModel):
    """Stable Audio Open 1.0 ComfyUI Repackaged model loader using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.STABLE_AUDIO_OPEN_1_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.STABLE_AUDIO_OPEN_1_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipeline = None
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="STABLE_AUDIO_OPEN_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.float32) -> StableAudioPipeline:
        """Load pipeline from single-file safetensors."""
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="stable-audio-open-1.0.safetensors",
        )

        self._pipeline = StableAudioPipeline.from_single_file(
            model_path,
            config=ORIGINAL_CONFIG_REPO,
            torch_dtype=dtype,
        )
        return self._pipeline

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Stable Audio DiT transformer model.

        Returns:
            StableAudioDiTModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._pipeline is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self._pipeline = self._pipeline.to(dtype=dtype_override)

        self._transformer = self._pipeline.transformer
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the StableAudioDiTModel transformer.

        Returns a dictionary of tensors matching the transformer's forward signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)
        seq_length = kwargs.get("seq_length", 8)

        hidden_states = torch.randn(batch_size, seq_length, HIDDEN_SIZE, dtype=dtype)
        timestep = torch.rand(batch_size, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, seq_length, HIDDEN_SIZE, dtype=dtype
        )
        global_hidden_states = torch.randn(batch_size, 1, HIDDEN_SIZE, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "global_hidden_states": global_hidden_states,
        }
