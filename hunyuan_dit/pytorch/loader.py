# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanDiT model loader implementation for text-to-image generation
"""
import torch
from diffusers import HunyuanDiT2DModel
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
    """Available HunyuanDiT model variants."""

    V1_1_DISTILLED = "v1.1 Distilled"


class ModelLoader(ForgeModel):
    """HunyuanDiT model loader implementation for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.V1_1_DISTILLED: ModelConfig(
            pretrained_model_name="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_1_DISTILLED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HunyuanDiT",
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

        self.transformer = HunyuanDiT2DModel.from_pretrained(
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

        # Latent dimensions
        sample_size = config.sample_size  # 128
        in_channels = config.in_channels  # 4
        hidden_states = torch.randn(
            batch_size, in_channels, sample_size, sample_size, dtype=dtype
        )

        # Timestep
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        # CLIP text encoder hidden states (text_len=77, cross_attention_dim=1024)
        encoder_hidden_states = torch.randn(
            batch_size, config.text_len, config.cross_attention_dim, dtype=dtype
        )

        # T5 text encoder hidden states (text_len_t5=256, cross_attention_dim_t5=2048)
        encoder_hidden_states_t5 = torch.randn(
            batch_size,
            config.text_len_t5,
            config.cross_attention_dim_t5,
            dtype=dtype,
        )

        # Image rotary embeddings (freqs_cis_img)
        # HunyuanDiT uses rotary position embeddings for spatial dimensions
        head_dim = config.attention_head_dim  # 88
        latent_h = sample_size // config.patch_size  # 128 // 2 = 64
        latent_w = sample_size // config.patch_size  # 128 // 2 = 64
        seq_len = latent_h * latent_w
        image_rotary_emb = (
            torch.randn(seq_len, head_dim // 2, dtype=dtype),
            torch.randn(seq_len, head_dim // 2, dtype=dtype),
        )

        # Pooled CLIP text embeddings (pooled_projection_dim=1024)
        text_embedding_mask = torch.ones(batch_size, config.text_len, dtype=dtype)
        encoder_attention_mask = torch.ones(batch_size, config.text_len_t5, dtype=dtype)

        # Style conditioning (optional, defaults to zeros)
        style = torch.zeros(batch_size, dtype=torch.long)

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embedding_mask": text_embedding_mask,
            "encoder_hidden_states_t5": encoder_hidden_states_t5,
            "encoder_attention_mask": encoder_attention_mask,
            "image_rotary_emb": image_rotary_emb,
            "style": style,
        }

        return inputs
