# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 ControlNet Fun Union model loader implementation for conditional image generation.

Loads the FLUX.2 ControlNet Fun Union model from alibaba-pai, which supports
multiple control conditions (Canny, HED, Depth, Pose, MLSD, Scribble, Gray)
in a single unified model.

Repository: https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union
"""

import torch
from diffusers.models import Flux2Transformer2DModel
from typing import Optional

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

REPO_ID = "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union"
SAFETENSORS_FILE = "FLUX.2-dev-Fun-Controlnet-Union.safetensors"


class ModelVariant(StrEnum):
    """Available FLUX.2 ControlNet Fun Union model variants."""

    CONTROLNET_UNION = "ControlNet_Union"


class ModelLoader(ForgeModel):
    """FLUX.2 ControlNet Fun Union model loader for conditional image generation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_UNION: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_UNION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None
        self.guidance_scale = 4.0

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FLUX.2 ControlNet Fun",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX.2 ControlNet Fun Union transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The FLUX.2 transformer model instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        repo_id = self._variant_config.pretrained_model_name
        self.transformer = Flux2Transformer2DModel.from_single_file(
            f"https://huggingface.co/{repo_id}/resolve/main/{SAFETENSORS_FILE}",
            torch_dtype=compute_dtype,
        )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FLUX.2 ControlNet Fun Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size (default: 1).

        Returns:
            dict: Input tensors for the transformer model.
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Image dimensions
        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        # Prepare latents: VAE compresses by vae_scale_factor, then pack 2x2 patches
        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        # Create latent tensor (B, C, H, W) then pack to (B, H*W, C)
        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )

        # Prepare latent image IDs (B, H*W, 4)
        t = torch.arange(1)
        h = torch.arange(h_packed)
        w = torch.arange(w_packed)
        l = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)

        # Pack latents: (B, C, H, W) -> (B, H*W, C)
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        # Prompt embeddings: use random tensors matching joint_attention_dim
        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        # Text IDs (B, seq_len, 4)
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(max_sequence_length)
        text_ids = torch.cartesian_prod(t, h, w, l)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)

        # Guidance
        guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        inputs = {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
