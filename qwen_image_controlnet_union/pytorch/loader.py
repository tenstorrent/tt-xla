# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image ControlNet Union model loader implementation for conditional image generation.

Loads the Qwen-Image ControlNet Union model from InstantX, which supports
multiple control conditions (Canny, Soft Edge, Depth, Pose) in a single
unified model based on the Qwen-Image diffusion transformer architecture.

Repository: https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union
"""

import torch
from diffusers import QwenImageControlNetModel
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

CONTROLNET_REPO_ID = "InstantX/Qwen-Image-ControlNet-Union"


class ModelVariant(StrEnum):
    """Available Qwen-Image ControlNet Union model variants."""

    CONTROLNET_UNION = "ControlNet_Union"


class ModelLoader(ForgeModel):
    """Qwen-Image ControlNet Union model loader for conditional image generation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_UNION: ModelConfig(
            pretrained_model_name=CONTROLNET_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_UNION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen-Image ControlNet Union",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen-Image ControlNet Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen-Image ControlNet model instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        repo_id = self._variant_config.pretrained_model_name
        self.controlnet = QwenImageControlNetModel.from_pretrained(
            repo_id, torch_dtype=compute_dtype
        )

        return self.controlnet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen-Image ControlNet Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size (default: 1).

        Returns:
            dict: Input tensors for the ControlNet model.
        """
        if self.controlnet is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.controlnet.config

        # Model config: num_attention_heads=24, attention_head_dim=128,
        # joint_attention_dim=3584, in_channels=64, patch_size=2,
        # axes_dims_rope=[16, 56, 56]
        hidden_size = config.num_attention_heads * config.attention_head_dim  # 3072
        joint_attention_dim = config.joint_attention_dim  # 3584

        # Image dimensions (small for testing)
        height = 128
        width = 128
        vae_scale_factor = 8
        patch_size = config.patch_size  # 2

        # Latent dimensions after VAE encoding and patchification
        h_latent = height // vae_scale_factor  # 16
        w_latent = width // vae_scale_factor  # 16
        h_patched = h_latent // patch_size  # 8
        w_patched = w_latent // patch_size  # 8
        seq_len = h_patched * w_patched  # 64

        # Hidden states (packed latent representation)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

        # Encoder hidden states (text embeddings)
        text_seq_len = 128
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )

        # Timestep
        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        # ControlNet conditioning image (processed through VAE-like encoder)
        controlnet_cond = torch.randn(
            batch_size, config.in_channels, h_latent, w_latent, dtype=dtype
        )

        # Image and text positional IDs for 3D RoPE
        # Image IDs: (batch, seq_len, 3) for [temporal, height, width]
        img_ids = torch.zeros(batch_size, seq_len, 3, dtype=dtype)
        for i in range(h_patched):
            for j in range(w_patched):
                idx = i * w_patched + j
                img_ids[:, idx, 1] = i
                img_ids[:, idx, 2] = j

        # Text IDs: (batch, text_seq_len, 3)
        txt_ids = torch.zeros(batch_size, text_seq_len, 3, dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "conditioning_scale": 1.0,
        }

        return inputs
