# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kandinsky 2.1 UNet model loader implementation.

Extracts the UNet2DConditionModel decoder from the Kandinsky 2.1 pipeline
for direct inference testing with synthetic tensor inputs.
"""

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from diffusers import UNet2DConditionModel


class ModelVariant(StrEnum):
    """Available Kandinsky 2.1 model variants."""

    V2_1 = "2.1"


class ModelLoader(ForgeModel):
    """Kandinsky 2.1 UNet model loader implementation."""

    _VARIANTS = {
        ModelVariant.V2_1: ModelConfig(
            pretrained_model_name="kandinsky-community/kandinsky-2-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Kandinsky",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kandinsky 2.1 decoder UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DConditionModel: The pre-trained Kandinsky 2.1 decoder UNet.
        """
        dtype = dtype_override or torch.float32
        unet = UNet2DConditionModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="unet",
            torch_dtype=dtype,
            **kwargs,
        )
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return synthetic tensor inputs for the Kandinsky 2.1 UNet.

        The UNet expects:
        - sample: noised latent (batch, in_channels=4, height=64, width=64)
        - timestep: diffusion timestep
        - encoder_hidden_states: text encoding (batch, seq_len=77, encoder_hid_dim=1024)
        - added_cond_kwargs: text_embeds and image_embeds (batch, cross_attn_dim=768)

        Args:
            dtype_override: Optional torch.dtype for input tensors.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Dictionary of input tensors for the UNet forward pass.
        """
        dtype = dtype_override or torch.float32
        return {
            "sample": torch.randn(batch_size, 4, 64, 64, dtype=dtype),
            "timestep": torch.tensor([0]),
            "encoder_hidden_states": torch.randn(batch_size, 77, 1024, dtype=dtype),
            "added_cond_kwargs": {
                "text_embeds": torch.randn(batch_size, 768, dtype=dtype),
                "image_embeds": torch.randn(batch_size, 768, dtype=dtype),
            },
        }
