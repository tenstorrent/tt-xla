# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Fun Control GGUF model loader implementation for video generation
"""
import torch
from diffusers import WanTransformer3DModel
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


class ModelVariant(StrEnum):
    """Available Wan 2.2 Fun Control GGUF model variants."""

    A14B_HIGHNOISE_Q4_K_M = "A14B_HighNoise_Q4_K_M"


class ModelLoader(ForgeModel):
    """Wan 2.2 Fun Control GGUF model loader for video generation tasks."""

    _VARIANTS = {
        ModelVariant.A14B_HIGHNOISE_Q4_K_M: ModelConfig(
            pretrained_model_name="QuantStack/Wan2.2-Fun-A14B-Control-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.A14B_HIGHNOISE_Q4_K_M

    GGUF_FILE = "HighNoise/Wan2.2-Fun-A14B-Control_HighNoise-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wan 2.2 Fun Control GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        load_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        self.transformer = WanTransformer3DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            **load_kwargs,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Video dimensions: (batch, channels, frames, height, width)
        num_frames = 1
        height = 32
        width = 32
        in_channels = config.in_channels

        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=dtype
        )

        # Timestep
        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)

        # Text encoder hidden states
        text_dim = config.text_dim
        seq_len = 64
        encoder_hidden_states = torch.randn(batch_size, seq_len, text_dim, dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        return inputs
