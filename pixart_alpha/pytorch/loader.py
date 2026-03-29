# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PixArt-Alpha model loader implementation for text-to-image generation
"""
import torch
from diffusers import PixArtTransformer2DModel
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


class ModelVariant(StrEnum):
    """Available PixArt-Alpha model variants."""

    XL_2_1024_MS = "XL-2-1024-MS"


class ModelLoader(ForgeModel):
    """PixArt-Alpha model loader implementation for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.XL_2_1024_MS: ModelConfig(
            pretrained_model_name="PixArt-alpha/PixArt-XL-2-1024-MS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XL_2_1024_MS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PixArt-Alpha",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        load_kwargs = {"use_safetensors": True}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        self.transformer = PixArtTransformer2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="transformer",
            **load_kwargs,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        config = self.transformer.config

        # Image dimensions for latent space
        sample_size = config.sample_size  # 128 for 1024px model
        in_channels = config.in_channels  # 4
        num_attention_heads = config.num_attention_heads  # 16
        attention_head_dim = config.attention_head_dim  # 72
        inner_dim = num_attention_heads * attention_head_dim  # 1152
        cross_attention_dim = config.cross_attention_dim  # 1152

        # Latent input: (B, C, H, W) where H,W = sample_size
        hidden_states = torch.randn(
            batch_size, in_channels, sample_size, sample_size, dtype=dtype
        )

        # T5 encoder hidden states: (B, seq_len, cross_attention_dim)
        max_sequence_length = 120
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, cross_attention_dim, dtype=dtype
        )

        # Timestep
        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)

        # Encoder attention mask: (B, seq_len)
        encoder_attention_mask = torch.ones(
            batch_size, max_sequence_length, dtype=dtype
        )

        # Added condition kwargs for resolution/aspect ratio
        resolution = torch.tensor([1024.0], dtype=dtype).expand(batch_size)
        aspect_ratio = torch.tensor([1.0], dtype=dtype).expand(batch_size)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "added_cond_kwargs": added_cond_kwargs,
            "encoder_attention_mask": encoder_attention_mask,
        }

        return inputs
