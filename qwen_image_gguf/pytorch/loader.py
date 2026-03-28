# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image GGUF model loader implementation for text-to-image generation.

Repository:
- https://huggingface.co/city96/Qwen-Image-gguf
"""
import torch
from diffusers import QwenImageTransformer2DModel
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

GGUF_BASE_URL = "https://huggingface.co/city96/Qwen-Image-gguf/blob/main"


class ModelVariant(StrEnum):
    """Available Qwen-Image GGUF model variants."""

    Q4_K_S = "Q4_K_S"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """Qwen-Image GGUF model loader implementation for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.Q4_K_S: ModelConfig(
            pretrained_model_name="city96/Qwen-Image-gguf",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="city96/Qwen-Image-gguf",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_K_S: "qwen-image-Q4_K_S.gguf",
        ModelVariant.Q8_0: "qwen-image-Q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen-Image GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        gguf_file = self._GGUF_FILES[self._variant]
        gguf_url = f"{GGUF_BASE_URL}/{gguf_file}"

        load_kwargs = {}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        self.transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_url,
            **load_kwargs,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Image dimensions
        height = 128
        width = 128
        patch_size = config.patch_size
        in_channels = config.in_channels

        # Compute latent sequence length: (H / patch) * (W / patch)
        h_patches = height // patch_size
        w_patches = width // patch_size
        image_seq_len = h_patches * w_patches

        # Hidden states: (batch, image_seq_len, in_channels)
        hidden_states = torch.randn(batch_size, image_seq_len, in_channels, dtype=dtype)

        # Encoder hidden states (text embeddings): (batch, text_seq_len, joint_attention_dim)
        text_seq_len = 128
        joint_attention_dim = config.joint_attention_dim
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )

        # Encoder hidden states mask: (batch, text_seq_len)
        encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        # Image shapes for RoPE: list of (frames, height, width)
        img_shapes = [(1, h_patches, w_patches)] * batch_size

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }

        return inputs
