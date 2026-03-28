# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FireRed-Image-Edit 1.1 ComfyUI Repackaged model loader implementation.

Loads a single-file safetensors diffusion transformer from
FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI. Uses the upstream
FireRedTeam/FireRed-Image-Edit-1.1 diffusers config for model construction.

Available variants:
- FIRERED_IMAGE_EDIT_1_1: FireRed-Image-Edit 1.1 transformer (bf16)
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

REPO_ID = "FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI"

# Diffusion model file path within the ComfyUI repackaged repo
_DIFFUSION_FILE = "FireRed-Image-Edit-1.1-transformer.safetensors"

# Upstream diffusers config source
_CONFIG_REPO = "FireRedTeam/FireRed-Image-Edit-1.1"


class ModelVariant(StrEnum):
    """Available FireRed-Image-Edit ComfyUI model variants."""

    FIRERED_IMAGE_EDIT_1_1 = "Edit_1.1"


class ModelLoader(ForgeModel):
    """FireRed-Image-Edit 1.1 ComfyUI model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.FIRERED_IMAGE_EDIT_1_1: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FIRERED_IMAGE_EDIT_1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIRERED_IMAGE_EDIT_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.float32
    ) -> QwenImageTransformer2DModel:
        """Load diffusion transformer from single-file safetensors."""
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_DIFFUSION_FILE,
        )

        self._transformer = QwenImageTransformer2DModel.from_single_file(
            model_path,
            config=_CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the FireRed-Image-Edit diffusion transformer.

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
