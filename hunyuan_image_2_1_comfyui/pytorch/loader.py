# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanImage 2.1 ComfyUI model loader implementation.

Loads single-file safetensors transformer from Comfy-Org/HunyuanImage_2.1_ComfyUI.
Supports the distilled variant for faster inference.

Available variants:
- DISTILLED_BF16: HunyuanImage 2.1 distilled transformer (bf16)
"""

from typing import Optional

import torch
from diffusers import HunyuanImageTransformer2DModel  # type: ignore[import]
from huggingface_hub import hf_hub_download  # type: ignore[import]

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

REPO_ID = "Comfy-Org/HunyuanImage_2.1_ComfyUI"


class ModelVariant(StrEnum):
    """Available HunyuanImage 2.1 ComfyUI model variants."""

    DISTILLED_BF16 = "Distilled_BF16"


class ModelLoader(ForgeModel):
    """HunyuanImage 2.1 ComfyUI model loader using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.DISTILLED_BF16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DISTILLED_BF16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HUNYUAN_IMAGE_2_1_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the HunyuanImage 2.1 transformer model.

        Returns:
            HunyuanImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="split_files/diffusion_models/hunyuanimage2.1_distilled_bf16.safetensors",
        )

        self._transformer = HunyuanImageTransformer2DModel.from_single_file(
            model_path,
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the HunyuanImage 2.1 transformer.

        Returns:
            dict: Input tensors matching the transformer's forward signature.
        """
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self._transformer.config

        # Image latent dimensions (small for testing)
        height = 16
        width = 16
        in_channels = config.in_channels  # 64

        # hidden_states: [batch, channels, height, width]
        hidden_states = torch.randn(batch_size, in_channels, height, width, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        # Text encoder hidden states (Qwen 2.5 VL 7B: text_embed_dim=3584)
        text_seq_len = 128
        text_embed_dim = config.text_embed_dim  # 3584
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, text_embed_dim, dtype=dtype
        )
        encoder_attention_mask = torch.ones(batch_size, text_seq_len, dtype=torch.bool)

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }

        return inputs
