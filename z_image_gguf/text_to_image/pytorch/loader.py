# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image GGUF model loader implementation for text-to-image generation.

Loads GGUF-quantized variants of the Z-Image-Turbo diffusion model
from gguf-org/z-image-gguf.
"""

from typing import Optional

import torch
from diffusers import AutoencoderKL, FluxTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

GGUF_REPO_ID = "gguf-org/z-image-gguf"

# VAE latent dimensions for testing
LATENT_CHANNELS = 16
LATENT_HEIGHT = 8
LATENT_WIDTH = 8


class ModelVariant(StrEnum):
    """Available Z-Image GGUF model variants."""

    Z_IMAGE_GGUF_VAE = "VAE_F16"
    Z_IMAGE_GGUF_Q4_K_M = "Q4_K_M"


_GGUF_FILES = {
    ModelVariant.Z_IMAGE_GGUF_VAE: "pig_flux_vae_fp32-f16.gguf",
    ModelVariant.Z_IMAGE_GGUF_Q4_K_M: "z-image-turbo-q4_k_m.gguf",
}


class ModelLoader(ForgeModel):
    """Z-Image GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.Z_IMAGE_GGUF_VAE: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
        ModelVariant.Z_IMAGE_GGUF_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Z_IMAGE_GGUF_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_IMAGE_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return _GGUF_FILES[self._variant]

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKL:
        """Load VAE from GGUF file."""
        vae_path = hf_hub_download(
            repo_id=GGUF_REPO_ID,
            filename=_GGUF_FILES[ModelVariant.Z_IMAGE_GGUF_VAE],
        )
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        self._vae = AutoencoderKL.from_single_file(
            vae_path,
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the model for the selected variant.

        For Z_IMAGE_GGUF_VAE: returns AutoencoderKL instance.
        For Z_IMAGE_GGUF_Q4_K_M: returns the quantized diffusion transformer.
        """
        if self._variant == ModelVariant.Z_IMAGE_GGUF_VAE:
            dtype = dtype_override if dtype_override is not None else torch.float32
            if self._vae is None:
                return self._load_vae(dtype)
            if dtype_override is not None:
                self._vae = self._vae.to(dtype=dtype_override)
            return self._vae

        # Transformer variant
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_path = hf_hub_download(
            repo_id=GGUF_REPO_ID,
            filename=self._gguf_file,
        )
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        transformer.eval()
        return transformer

    def load_inputs(self, **kwargs):
        """Prepare inputs for the selected variant.

        For Z_IMAGE_GGUF_VAE: returns random latent tensor for decoding.
        For Z_IMAGE_GGUF_Q4_K_M: returns random latent inputs for the transformer.
        """
        if self._variant == ModelVariant.Z_IMAGE_GGUF_VAE:
            return self._load_vae_inputs(**kwargs)
        return self._load_transformer_inputs(**kwargs)

    def _load_vae_inputs(self, **kwargs):
        """Prepare inputs for the VAE variant."""
        dtype = kwargs.get("dtype_override", torch.float32)
        return torch.randn(
            1,
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            dtype=dtype,
        )

    def _load_transformer_inputs(self, **kwargs):
        """Prepare random inputs for the transformer variant."""
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)
        height = 128
        width = 128

        # Approximate latent dimensions
        latent_h = height // 16
        latent_w = width // 16
        num_channels = 64

        hidden_states = torch.randn(
            batch_size, latent_h * latent_w, num_channels, dtype=dtype
        )
        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)
        # Dummy encoder hidden states (text embeddings)
        encoder_hidden_states = torch.randn(batch_size, 128, 4096, dtype=dtype)
        txt_ids = torch.zeros(batch_size, 128, 3, dtype=dtype)
        img_ids = torch.zeros(batch_size, latent_h * latent_w, 3, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
        }
