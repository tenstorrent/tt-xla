# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VanillaMotionBotModels Wan 2.2 I2V GGUF model loader implementation for video generation
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
    """Available VanillaMotionBotModels variants."""

    I2V_14B_HIGHNOISE_Q8_0 = "I2V_14B_HighNoise_Q8_0"
    I2V_14B_LOWNOISE_Q8_0 = "I2V_14B_LowNoise_Q8_0"


class ModelLoader(ForgeModel):
    """VanillaMotionBotModels Wan 2.2 I2V GGUF model loader for video generation tasks."""

    _VARIANTS = {
        ModelVariant.I2V_14B_HIGHNOISE_Q8_0: ModelConfig(
            pretrained_model_name="matadamovic/vanillamotionbotmodels",
        ),
        ModelVariant.I2V_14B_LOWNOISE_Q8_0: ModelConfig(
            pretrained_model_name="matadamovic/vanillamotionbotmodels",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.I2V_14B_HIGHNOISE_Q8_0

    _GGUF_FILES = {
        ModelVariant.I2V_14B_HIGHNOISE_Q8_0: "checkpoints_gguf/wan2.2_i2v_high_noise_14B_Q8_0.gguf",
        ModelVariant.I2V_14B_LOWNOISE_Q8_0: "checkpoints_gguf/wan2.2_i2v_low_noise_14B_Q8_0.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="VanillaMotionBotModels",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        gguf_file = self._GGUF_FILES[self._variant]
        load_kwargs = {"gguf_file": gguf_file}
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
