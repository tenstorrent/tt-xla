# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
smoothMixWan22 I2V GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 image-to-video diffusion transformer variants from
Bedovyy/smoothMixWan22-I2V-GGUF. Uses the upstream Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
config for model construction.

Available variants:
- HIGH_NOISE_Q4_K_M: HighNoise 4-bit quantization (medium)
- LOW_NOISE_Q4_K_M: LowNoise 4-bit quantization (medium)
"""

from typing import Any, Optional

import torch
from diffusers import WanTransformer3DModel
from huggingface_hub import hf_hub_download

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

REPO_ID = "Bedovyy/smoothMixWan22-I2V-GGUF"
CONFIG_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

_GGUF_FILES = {
    "HIGH_NOISE_Q4_K_M": "smoothMixWan22I2VT2V_i2vHigh-Q4_K_M.gguf",
    "LOW_NOISE_Q4_K_M": "smoothMixWan22I2VT2V_i2vLow-Q4_K_M.gguf",
}


class ModelVariant(StrEnum):
    """Available smoothMixWan22 I2V GGUF quantization variants."""

    HIGH_NOISE_Q4_K_M = "HighNoise_Q4_K_M"
    LOW_NOISE_Q4_K_M = "LowNoise_Q4_K_M"


class ModelLoader(ForgeModel):
    """smoothMixWan22 I2V GGUF model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.HIGH_NOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.LOW_NOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HIGH_NOISE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SMOOTH_MIX_WAN22_I2V_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_quant_key(self) -> str:
        """Map variant to quantization key for file lookup."""
        return {
            ModelVariant.HIGH_NOISE_Q4_K_M: "HIGH_NOISE_Q4_K_M",
            ModelVariant.LOW_NOISE_Q4_K_M: "LOW_NOISE_Q4_K_M",
        }[self._variant]

    def _load_transformer(
        self, dtype: torch.dtype = torch.float32
    ) -> WanTransformer3DModel:
        """Load diffusion transformer from GGUF file."""
        quant_key = self._get_quant_key()

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_GGUF_FILES[quant_key],
        )

        self._transformer = WanTransformer3DModel.from_single_file(
            model_path,
            config=CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the smoothMixWan22 I2V GGUF diffusion transformer."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the Wan I2V diffusion transformer."""
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        # Wan I2V 14B transformer config dimensions
        in_channels = 36  # 16 latent + 16 latent + 4 mask channels for I2V
        text_dim = 4096  # text_dim from Wan config
        txt_seq_len = 32

        # Spatial/temporal latent dimensions
        frame, height, width = 2, 8, 8
        seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, seq_len, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }
