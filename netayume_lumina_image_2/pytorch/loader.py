# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NetaYume Lumina Image 2.0 model loader implementation for text-to-image generation.

Loads the Lumina2Transformer2DModel from duongve/NetaYume-Lumina-Image-2.0,
an anime-focused text-to-image diffusion model fine-tuned from Alpha-VLLM/Lumina-Image-2.0.
Uses Gemma-2-2b text encoder and Flux.1 VAE.
"""

import torch
from diffusers import Lumina2Transformer2DModel
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

REPO_ID = "duongve/NetaYume-Lumina-Image-2.0"
SINGLE_FILE = "Unet/v2/NetaYume_Lumina_v2_unet.safetensors"

# Lumina-Image-2.0 architecture constants
IN_CHANNELS = 16
HIDDEN_SIZE = 2304
CAP_FEAT_DIM = 2304
PATCH_SIZE = 2


class ModelVariant(StrEnum):
    """Available NetaYume Lumina Image 2.0 model variants."""

    V2 = "v2"


class ModelLoader(ForgeModel):
    """NetaYume Lumina Image 2.0 model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.V2: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="NetaYume_Lumina_Image_2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        load_kwargs = {}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        self.transformer = Lumina2Transformer2DModel.from_single_file(
            f"https://huggingface.co/{REPO_ID}/blob/main/{SINGLE_FILE}",
            **load_kwargs,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        self.transformer.eval()
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32

        # Latent image: (B, in_channels, H, W)
        height = 128
        width = 128
        hidden_states = torch.randn(batch_size, IN_CHANNELS, height, width, dtype=dtype)

        # Timestep: (B,)
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        # Text encoder hidden states from Gemma-2-2b: (B, seq_len, cap_feat_dim)
        max_sequence_length = 128
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, CAP_FEAT_DIM, dtype=dtype
        )

        # Encoder attention mask: (B, seq_len)
        encoder_attention_mask = torch.ones(
            batch_size, max_sequence_length, dtype=torch.bool
        )

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }

        return inputs
