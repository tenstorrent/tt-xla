# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Image model loader implementation.

Loads the diffusion transformer from Comfy-Org/LongCat-Image single-file
safetensors checkpoint. This is a Flux-based architecture without guidance
embedding support.

Available variants:
- LONGCAT_IMAGE_BF16: LongCat-Image bf16 diffusion transformer
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

REPO_ID = "Comfy-Org/LongCat-Image"
SAFETENSORS_FILE = "split_files/diffusion_models/longcat_image_bf16.safetensors"


class ModelVariant(StrEnum):
    """Available LongCat-Image model variants."""

    LONGCAT_IMAGE_BF16 = "bf16"


class ModelLoader(ForgeModel):
    """LongCat-Image model loader for the Flux-based diffusion transformer."""

    _VARIANTS = {
        ModelVariant.LONGCAT_IMAGE_BF16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LONGCAT_IMAGE_BF16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LONGCAT_IMAGE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> FluxTransformer2DModel:
        """Load the diffusion transformer from the single-file checkpoint."""
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=SAFETENSORS_FILE,
        )

        self._transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the LongCat-Image diffusion transformer.

        Returns:
            FluxTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns:
            dict: Input tensors matching FluxTransformer2DModel forward signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        if self._transformer is None:
            self.load_model(dtype_override=dtype)

        config = self._transformer.config

        # Image dimensions
        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        # Prepare packed latents: (B, H*W, C)
        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        # Latent image IDs (seq_len, 3)
        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Text embeddings matching joint_attention_dim
        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        # Text IDs (seq_len, 3)
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
