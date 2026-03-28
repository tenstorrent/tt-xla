# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-Rapid-AIO GGUF model loader implementation.

Loads GGUF-quantized diffusion transformer variants from
Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF. Uses the upstream Qwen/Qwen-Image-Edit-2511
diffusers config for model construction.

Available variants:
- QWEN_IMAGE_EDIT_RAPID_AIO_Q4_K_M: Q4_K_M quantization (13.1 GB)
- QWEN_IMAGE_EDIT_RAPID_AIO_Q8_0: Q8_0 quantization (21.8 GB)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageTransformer2DModel
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

REPO_ID = "Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF"

# Upstream diffusers config for model construction
CONFIG_REPO = "Qwen/Qwen-Image-Edit-2511"

# GGUF quantization files within the repo
_GGUF_FILES = {
    "Q4_K_M": "qwen-image-edit-rapid-aio-Q4_K_M.gguf",
    "Q8_0": "qwen-image-edit-rapid-aio-Q8_0.gguf",
}


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-Rapid-AIO GGUF model variants."""

    QWEN_IMAGE_EDIT_RAPID_AIO_Q4_K_M = "Rapid_AIO_Q4_K_M"
    QWEN_IMAGE_EDIT_RAPID_AIO_Q8_0 = "Rapid_AIO_Q8_0"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-Rapid-AIO GGUF model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_RAPID_AIO_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.QWEN_IMAGE_EDIT_RAPID_AIO_Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_RAPID_AIO_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_RAPID_AIO_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_quantization_key(self) -> str:
        """Map variant to quantization key for GGUF file lookup."""
        return {
            ModelVariant.QWEN_IMAGE_EDIT_RAPID_AIO_Q4_K_M: "Q4_K_M",
            ModelVariant.QWEN_IMAGE_EDIT_RAPID_AIO_Q8_0: "Q8_0",
        }[self._variant]

    def _load_transformer(
        self, dtype: torch.dtype = torch.float32
    ) -> QwenImageTransformer2DModel:
        """Load diffusion transformer from GGUF file."""
        quant_key = self._get_quantization_key()

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_GGUF_FILES[quant_key],
        )

        self._transformer = QwenImageTransformer2DModel.from_single_file(
            model_path,
            config=CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Qwen-Image-Edit-Rapid-AIO diffusion transformer.

        Returns:
            QwenImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        # From model config: in_channels=64 (img_in linear input dimension)
        img_dim = 64
        # joint_attention_dim from config = 3584
        text_dim = 3584
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
