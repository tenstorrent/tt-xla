# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXL-Turbo Ryzen AI (stabilityai/sdxl-turbo-ryzen-ai) ONNX model loader.

This is an ONNX variant of SDXL-Turbo optimized for AMD Ryzen AI NPUs using
Block FP16 quantization. The loader downloads and loads the UNet ONNX model
component from the HuggingFace repository.

Available variants:
- SDXL_TURBO_RYZEN_AI: stabilityai/sdxl-turbo-ryzen-ai UNet ONNX model
"""

import onnx
import torch
from typing import Optional
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


REPO_ID = "stabilityai/sdxl-turbo-ryzen-ai"


class ModelVariant(StrEnum):
    """Available SDXL-Turbo Ryzen AI model variants."""

    SDXL_TURBO_RYZEN_AI = "SDXL_Turbo_Ryzen_AI"


class ModelLoader(ForgeModel):
    """SDXL-Turbo Ryzen AI ONNX model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXL_TURBO_RYZEN_AI: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SDXL_TURBO_RYZEN_AI

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXL_Turbo_Ryzen_AI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the SDXL-Turbo Ryzen AI UNet ONNX model.

        Downloads the UNet ONNX model from HuggingFace and loads it.

        Returns:
            onnx.ModelProto: The UNet ONNX model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(pretrained_model_name, filename="unet/model.onnx")
        model = onnx.load(model_path)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the SDXL-Turbo Ryzen AI UNet model.

        Generates random latent noise, timestep, and encoder hidden states
        matching the expected UNet input shapes.

        Returns:
            dict: Dictionary of input tensors for the UNet model.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        # Latent noise input (batch, channels, height/8, width/8)
        # SDXL uses 4 latent channels, 512x512 image -> 64x64 latents
        sample = torch.randn(batch_size, 4, 64, 64, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0], dtype=dtype)

        # Encoder hidden states from dual CLIP text encoders
        # SDXL uses 2048-dim cross-attention from concatenated CLIP outputs
        encoder_hidden_states = torch.randn(batch_size, 77, 2048, dtype=dtype)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
