# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Image-Edit model loader implementation.

Loads the LongCat-Image-Edit-Turbo diffusion transformer for image editing.
Uses LongCatImageEditPipeline from diffusers with a Qwen2.5-VL text encoder.

Available variants:
- TURBO: LongCat-Image-Edit-Turbo (bf16, 8 NFEs)
"""

from typing import Any, Optional

import torch
from diffusers import LongCatImageEditPipeline

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

REPO_ID = "meituan-longcat/LongCat-Image-Edit-Turbo"

# From transformer/config.json
IN_CHANNELS = 64
JOINT_ATTENTION_DIM = 3584
POOLED_PROJECTION_DIM = 3584


class ModelVariant(StrEnum):
    """Available LongCat-Image-Edit model variants."""

    TURBO = "Turbo"


class ModelLoader(ForgeModel):
    """LongCat-Image-Edit model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.TURBO: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LONGCAT_IMAGE_EDIT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> LongCatImageEditPipeline:
        """Load the full LongCat-Image-Edit pipeline."""
        self._pipe = LongCatImageEditPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the LongCat-Image-Edit diffusion transformer.

        Returns:
            LongCatImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the LongCat diffusion transformer.

        Returns a dict matching LongCatImageTransformer2DModel.forward() signature.
        The transformer uses a FLUX-like architecture with joint and single attention
        layers, so the input format follows the same hidden_states / encoder_hidden_states
        pattern with positional IDs.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        txt_seq_len = 32
        # Image latent grid: patch_size=1, so latent pixels = spatial tokens
        latent_h = 16
        latent_w = 16
        img_seq_len = latent_h * latent_w

        # Hidden states: packed latents in sequence format (batch, img_seq, in_channels)
        hidden_states = torch.randn(batch_size, img_seq_len, IN_CHANNELS, dtype=dtype)

        # Text encoder outputs
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, JOINT_ATTENTION_DIM, dtype=dtype
        )
        pooled_projections = torch.randn(batch_size, POOLED_PROJECTION_DIM, dtype=dtype)

        # Timestep for diffusion
        timestep = torch.tensor([0.5] * batch_size, dtype=dtype)

        # Positional IDs: text IDs are zeros, image IDs encode spatial position
        txt_ids = torch.zeros(txt_seq_len, 3, dtype=dtype)
        img_ids = torch.zeros(latent_h, latent_w, 3, dtype=dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(latent_h)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(latent_w)[None, :]
        img_ids = img_ids.reshape(-1, 3)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
        }
