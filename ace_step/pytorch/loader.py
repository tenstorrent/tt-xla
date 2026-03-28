# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step model loader implementation for text-to-music generation tasks.
"""
import torch
import torch.nn as nn
from typing import Optional

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


class ACEStepWrapper(nn.Module):
    """Wrapper around ACE-Step transformer to expose a standard forward pass."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )


class ModelVariant(StrEnum):
    """Available ACE-Step model variants."""

    ACE_STEP_V1_3_5B = "v1-3.5B"


class ModelLoader(ForgeModel):
    """ACE-Step model loader implementation for text-to-music generation tasks."""

    _VARIANTS = {
        ModelVariant.ACE_STEP_V1_3_5B: ModelConfig(
            pretrained_model_name="ACE-Step/ACE-Step-v1-3.5B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ACE_STEP_V1_3_5B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ACE-Step",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from acestep.pipeline_ace_step import ACEStepPipeline

        dtype_str = "float32"
        if dtype_override == torch.bfloat16:
            dtype_str = "bfloat16"
        elif dtype_override == torch.float16:
            dtype_str = "float16"

        self.pipeline = ACEStepPipeline(
            checkpoint_dir="",
            dtype=dtype_str,
            torch_compile=False,
            cpu_offload=False,
        )

        transformer = self.pipeline.transformer
        model = ACEStepWrapper(transformer)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override or torch.float32
        batch_size = 1
        channels = self.pipeline.transformer.config.in_channels
        height = 16
        width = 16

        hidden_states = torch.randn(batch_size, channels, height, width, dtype=dtype)
        timestep = torch.tensor([500.0], dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size,
            32,
            self.pipeline.transformer.config.cross_attention_dim,
            dtype=dtype,
        )

        return [hidden_states, timestep, encoder_hidden_states]
