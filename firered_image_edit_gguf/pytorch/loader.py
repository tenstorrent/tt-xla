# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FireRed-Image-Edit-1.1 GGUF model loader implementation.

Loads GGUF-quantized QwenImageTransformer2DModel variants from
vantagewithai/FireRed-Image-Edit-1.1-GGUF. Based on the FireRedTeam/FireRed-Image-Edit-1.1
diffusion transformer for general-purpose image editing.

Available variants:
- Q4_K_M: Q4_K_M quantization (13.1 GB)
- Q8_0: Q8_0 quantization (21.8 GB)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageTransformer2DModel
from diffusers.quantizers import GGUFQuantizationConfig
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

REPO_ID = "vantagewithai/FireRed-Image-Edit-1.1-GGUF"

# Upstream diffusers config for model construction
CONFIG_REPO = "FireRedTeam/FireRed-Image-Edit-1.1"

# GGUF filenames keyed by variant
_GGUF_FILES = {
    "Q4_K_M": "FireRed-Image-Edit-1.1-Q4_K_M.gguf",
    "Q8_0": "FireRed-Image-Edit-1.1-Q8_0.gguf",
}

# From transformer/config.json
IN_CHANNELS = 64
JOINT_ATTENTION_DIM = 3584


class ModelVariant(StrEnum):
    """Available FireRed-Image-Edit GGUF model variants."""

    Q4_K_M = "Q4_K_M"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """FireRed-Image-Edit-1.1 GGUF model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIRERED_IMAGE_EDIT_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_gguf_file(self) -> str:
        """Get the GGUF filename for the current variant."""
        return _GGUF_FILES[self._variant.value]

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> QwenImageTransformer2DModel:
        """Load diffusion transformer from GGUF quantized file."""
        gguf_file = self._get_gguf_file()

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=gguf_file,
        )

        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        self._transformer = QwenImageTransformer2DModel.from_single_file(
            model_path,
            config=CONFIG_REPO,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the GGUF-quantized FireRed diffusion transformer.

        Returns:
            QwenImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        # From model config: in_channels=64 (img_in linear input dimension)
        img_dim = IN_CHANNELS
        # joint_attention_dim from config = 3584
        text_dim = JOINT_ATTENTION_DIM
        txt_seq_len = 32

        # img_seq_len must equal frame * height * width for positional encoding
        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        # img_shapes: list of (frame, height, width) tuples per batch item
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
