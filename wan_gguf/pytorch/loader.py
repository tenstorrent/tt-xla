# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 GGUF model loader implementation.

Loads GGUF-quantized WanTransformer3DModel variants from
MonsterMMORPG/Wan_GGUF. These are quantized versions of the Wan 2.1
image-to-video 14B diffusion transformer for efficient storage and inference.

Available variants:
- I2V_14B_720P_Q4_K_M: Q4_K_M quantized Wan 2.1 I2V 14B 720P transformer
"""

from typing import Any, Optional

import torch
from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
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

REPO_ID = "MonsterMMORPG/Wan_GGUF"

# Upstream diffusers config for model construction
CONFIG_REPO = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
TRANSFORMER_SUBFOLDER = "transformer"

# GGUF filenames keyed by variant
_GGUF_FILES = {
    "I2V_14B_720P_Q4_K_M": "wan2.1-i2v-14b-720p-Q4_K_M.gguf",
}

# WanTransformer3DModel input dimensions (from default config)
IN_CHANNELS = 36
TEXT_DIM = 4096
NUM_FRAMES = 9
LATENT_HEIGHT = 60
LATENT_WIDTH = 104


class ModelVariant(StrEnum):
    """Available Wan GGUF model variants."""

    I2V_14B_720P_Q4_K_M = "I2V_14B_720P_Q4_K_M"


class ModelLoader(ForgeModel):
    """Wan GGUF model loader for quantized Wan diffusion transformer."""

    _VARIANTS = {
        ModelVariant.I2V_14B_720P_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.I2V_14B_720P_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_gguf_file(self) -> str:
        """Get the GGUF filename for the current variant."""
        return _GGUF_FILES[self._variant.value]

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Wan transformer from GGUF format.

        Returns:
            WanTransformer3DModel instance loaded from GGUF checkpoint.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            gguf_file = self._get_gguf_file()
            gguf_path = hf_hub_download(repo_id=REPO_ID, filename=gguf_file)
            self._transformer = WanTransformer3DModel.from_single_file(
                gguf_path,
                config=CONFIG_REPO,
                subfolder=TRANSFORMER_SUBFOLDER,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            self._transformer.eval()
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare synthetic inputs for the Wan transformer.

        Returns:
            dict: Input tensors matching the WanTransformer3DModel forward signature:
                - hidden_states: Latent tensor [batch, channels, frames, height, width]
                - timestep: Scalar timestep tensor
                - encoder_hidden_states: Text encoder outputs [batch, seq_len, dim]
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)
        txt_seq_len = 512

        return {
            "hidden_states": torch.randn(
                batch_size,
                IN_CHANNELS,
                NUM_FRAMES,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            ),
            "timestep": torch.tensor([500] * batch_size, dtype=torch.long),
            "encoder_hidden_states": torch.randn(
                batch_size, txt_seq_len, TEXT_DIM, dtype=dtype
            ),
        }
