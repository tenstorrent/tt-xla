# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Krea-dev ComfyUI Repackaged model loader implementation.

Loads a single-file FP8 safetensors diffusion transformer from
Comfy-Org/FLUX.1-Krea-dev_ComfyUI. Uses the upstream black-forest-labs/FLUX.1-dev
diffusers config for model construction.

Reference: https://huggingface.co/Comfy-Org/FLUX.1-Krea-dev_ComfyUI
"""

from typing import Any, Optional

import torch
from diffusers import FluxTransformer2DModel
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

REPO_ID = "Comfy-Org/FLUX.1-Krea-dev_ComfyUI"

# Diffusion model file path within the ComfyUI repackaged repo
_DIFFUSION_FILE = "split_files/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors"

# Upstream diffusers config source
_CONFIG_REPO = "black-forest-labs/FLUX.1-dev"


class ModelVariant(StrEnum):
    """Available FLUX.1-Krea-dev ComfyUI model variants."""

    KREA_DEV_FP8 = "Krea_Dev_FP8"


class ModelLoader(ForgeModel):
    """FLUX.1-Krea-dev ComfyUI model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.KREA_DEV_FP8: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.KREA_DEV_FP8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX_KREA_DEV_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.float32
    ) -> FluxTransformer2DModel:
        """Load diffusion transformer from single-file safetensors."""
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_DIFFUSION_FILE,
        )

        self._transformer = FluxTransformer2DModel.from_single_file(
            model_path,
            config=_CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the FLUX.1-Krea-dev diffusion transformer.

        Returns:
            FluxTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the FLUX diffusion transformer.

        Returns a dict matching FluxTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        # FLUX.1-dev transformer config dimensions
        in_channels = 64  # patched latent input dimension
        pooled_projection_dim = 768  # CLIP pooled embedding dimension
        joint_attention_dim = 4096  # T5 encoder hidden dimension
        txt_seq_len = 256
        guidance_scale = 3.5

        # Image latent sequence: (height_latent // 2) * (width_latent // 2)
        # Using 128x128 image -> 16x16 latent -> 8x8 packed = 64 patches
        img_seq_len = 64

        hidden_states = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, joint_attention_dim, dtype=dtype
        )
        pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=dtype)
        timestep = torch.tensor([1.0] * batch_size, dtype=dtype)
        guidance = torch.full([batch_size], guidance_scale, dtype=dtype)

        # Text and image positional IDs (3D coordinates)
        txt_ids = torch.zeros(txt_seq_len, 3, dtype=dtype)
        img_ids = torch.zeros(img_seq_len, 3, dtype=dtype)
        # Fill spatial coordinates for image IDs
        h, w = 8, 8
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                img_ids[idx, 1] = float(i)
                img_ids[idx, 2] = float(j)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_projections,
            "encoder_hidden_states": encoder_hidden_states,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            "joint_attention_kwargs": {},
        }
