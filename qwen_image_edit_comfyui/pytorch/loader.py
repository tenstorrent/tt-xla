# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit ComfyUI Repackaged model loader implementation.

Loads single-file safetensors diffusion transformer variants from
Comfy-Org/Qwen-Image-Edit_ComfyUI. Uses the upstream Qwen/Qwen-Image-Edit
diffusers config for model construction.

Available variants:
- QWEN_IMAGE_EDIT_2509: Qwen-Image-Edit 2509 (bf16)
- QWEN_IMAGE_EDIT_2511: Qwen-Image-Edit 2511 (bf16)
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

REPO_ID = "Comfy-Org/Qwen-Image-Edit_ComfyUI"

# Diffusion model file paths within the ComfyUI repackaged repo
_DIFFUSION_FILES = {
    "2509": "split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors",
    "2511": "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
}

# Upstream diffusers config sources
_CONFIGS = {
    "2509": "Qwen/Qwen-Image-Edit-2509",
    "2511": "Qwen/Qwen-Image-Edit-2511",
}


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit ComfyUI model variants."""

    QWEN_IMAGE_EDIT_2509 = "Edit_2509"
    QWEN_IMAGE_EDIT_2511 = "Edit_2511"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit ComfyUI model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_2509: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.QWEN_IMAGE_EDIT_2511: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_2509

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_version_key(self) -> str:
        """Map variant to version key for file/config lookup."""
        return {
            ModelVariant.QWEN_IMAGE_EDIT_2509: "2509",
            ModelVariant.QWEN_IMAGE_EDIT_2511: "2511",
        }[self._variant]

    def _load_transformer(
        self, dtype: torch.dtype = torch.float32
    ) -> QwenImageTransformer2DModel:
        """Load diffusion transformer from single-file safetensors."""
        version = self._get_version_key()

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_DIFFUSION_FILES[version],
        )

        self._transformer = QwenImageTransformer2DModel.from_single_file(
            model_path,
            config=_CONFIGS[version],
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Qwen-Image-Edit diffusion transformer.

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
