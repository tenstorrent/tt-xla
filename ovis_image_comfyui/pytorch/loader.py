# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ovis-Image ComfyUI Repackaged model loader implementation.

Loads the single-file safetensors diffusion transformer from
Comfy-Org/Ovis-Image. Uses the upstream AIDC-AI/Ovis-Image-7B
diffusers config for model construction.

Available variants:
- OVIS_IMAGE: Ovis-Image diffusion transformer (bf16)
"""

from typing import Any, Optional

import torch
from diffusers import OvisImageTransformer2DModel
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

REPO_ID = "Comfy-Org/Ovis-Image"
UPSTREAM_REPO = "AIDC-AI/Ovis-Image-7B"

# Diffusion model file path within the ComfyUI repackaged repo
_DIFFUSION_FILE = "split_files/diffusion_models/ovis_image_bf16.safetensors"

# From transformer config: in_channels=64, joint_attention_dim=2048
IMG_DIM = 64
TEXT_DIM = 2048
TXT_SEQ_LEN = 32
# img_seq_len = frame * height * width for positional encoding
FRAME, HEIGHT, WIDTH = 1, 8, 8
IMG_SEQ_LEN = FRAME * HEIGHT * WIDTH


class ModelVariant(StrEnum):
    """Available Ovis-Image ComfyUI model variants."""

    OVIS_IMAGE = "Ovis_Image"


class ModelLoader(ForgeModel):
    """Ovis-Image ComfyUI model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.OVIS_IMAGE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.OVIS_IMAGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OVIS_IMAGE_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.float32
    ) -> OvisImageTransformer2DModel:
        """Load diffusion transformer from single-file safetensors."""
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_DIFFUSION_FILE,
        )

        self._transformer = OvisImageTransformer2DModel.from_single_file(
            model_path,
            config=UPSTREAM_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Ovis-Image diffusion transformer.

        Returns:
            OvisImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching OvisImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        hidden_states = torch.randn(batch_size, IMG_SEQ_LEN, IMG_DIM, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, TXT_SEQ_LEN, TEXT_DIM, dtype=dtype
        )
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)

        # Position IDs for image and text tokens (2D: [seq_len, 3])
        img_ids = torch.zeros(IMG_SEQ_LEN, 3, dtype=dtype)
        for i in range(IMG_SEQ_LEN):
            img_ids[i, 0] = 0  # frame
            img_ids[i, 1] = i // WIDTH  # height
            img_ids[i, 2] = i % WIDTH  # width

        txt_ids = torch.zeros(TXT_SEQ_LEN, 3, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }
