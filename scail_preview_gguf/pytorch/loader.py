# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SCAIL-Preview GGUF model loader implementation for image-to-video generation.

SCAIL (Studio-Grade Character Animation via In-Context Learning) animates
characters from a single image using 3D-consistent pose representations.
Based on the Wan 2.1 14B architecture in GGUF format for ComfyUI.

Repository:
- https://huggingface.co/vantagewithai/SCAIL-Preview-GGUF
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

GGUF_BASE_URL = "https://huggingface.co/vantagewithai/SCAIL-Preview-GGUF/blob/main"


class ModelVariant(StrEnum):
    """Available SCAIL-Preview GGUF model variants."""

    Q4_K_M = "Q4_K_M"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """SCAIL-Preview GGUF model loader for image-to-video character animation."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: ModelConfig(
            pretrained_model_name="vantagewithai/SCAIL-Preview-GGUF",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="vantagewithai/SCAIL-Preview-GGUF",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "Wan21-14B-SCAIL-preview_comfy-Q4_K_M.gguf",
        ModelVariant.Q8_0: "Wan21-14B-SCAIL-preview_comfy-Q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SCAIL-Preview GGUF",
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

        # Wan 2.1 video transformer dimensions
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
