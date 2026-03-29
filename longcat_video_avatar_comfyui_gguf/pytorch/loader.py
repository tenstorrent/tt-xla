# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Video-Avatar ComfyUI GGUF model loader implementation.

LongCat is an audio-driven character animation model that generates
expressive talking avatar videos. Based on the WAN 16B architecture
in GGUF format for ComfyUI, supporting both single-stream and
multi-stream audio inputs.

Repository:
- https://huggingface.co/vantagewithai/LongCat-Video-Avatar-ComfyUI-GGUF
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

GGUF_BASE_URL = (
    "https://huggingface.co/vantagewithai/LongCat-Video-Avatar-ComfyUI-GGUF"
    "/blob/main"
)


class ModelVariant(StrEnum):
    """Available LongCat-Video-Avatar ComfyUI GGUF model variants."""

    SINGLE_Q4_K_M = "Single_Q4_K_M"
    SINGLE_Q8_0 = "Single_Q8_0"


class ModelLoader(ForgeModel):
    """LongCat-Video-Avatar ComfyUI GGUF model loader for audio-driven avatar animation."""

    _VARIANTS = {
        ModelVariant.SINGLE_Q4_K_M: ModelConfig(
            pretrained_model_name="vantagewithai/LongCat-Video-Avatar-ComfyUI-GGUF",
        ),
        ModelVariant.SINGLE_Q8_0: ModelConfig(
            pretrained_model_name="vantagewithai/LongCat-Video-Avatar-ComfyUI-GGUF",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.SINGLE_Q4_K_M: "single/LongCat-Avatar-Single_comfy-Q4_K_M.gguf",
        ModelVariant.SINGLE_Q8_0: "single/LongCat-Avatar-Single_comfy-Q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.SINGLE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LongCat-Video-Avatar-ComfyUI GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        gguf_file = self._GGUF_FILES[self._variant]
        gguf_url = f"{GGUF_BASE_URL}/{gguf_file}"

        load_kwargs = {}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        self.transformer = WanTransformer3DModel.from_single_file(
            gguf_url,
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

        # WAN 16B video transformer dimensions
        num_channels = config.in_channels
        num_frames = 9
        height = 60  # latent height (480p / 8)
        width = 104  # latent width (832p / 8)

        # Latent video tensor: [batch, channels, frames, height, width]
        hidden_states = torch.randn(
            batch_size, num_channels, num_frames, height, width, dtype=dtype
        )

        # Timestep
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        # Text encoder hidden states
        encoder_hidden_states = torch.randn(
            batch_size, 256, config.text_dim, dtype=dtype
        )

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        return inputs
