# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
z_image_turbo model loader implementation.

Supports loading the VAE component from single-file safetensors and
the full Tongyi-MAI/Z-Image-Turbo text-to-image pipeline transformer.

Available variants:
- Z_IMAGE_TURBO_VAE: z_image_turbo VAE autoencoder (Comfy-Org/z_image_turbo)
- Z_IMAGE_TURBO: Full Z-Image-Turbo DiT transformer (Tongyi-MAI/Z-Image-Turbo)
"""

from typing import Any, Optional

import os

import torch
from diffusers import AutoencoderKL, ZImagePipeline
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

VAE_REPO_ID = "Comfy-Org/z_image_turbo"
PIPELINE_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"

# VAE latent dimensions for testing
LATENT_CHANNELS = 16
LATENT_HEIGHT = 8
LATENT_WIDTH = 8


class ModelVariant(StrEnum):
    """Available z_image_turbo model variants."""

    Z_IMAGE_TURBO_VAE = "VAE"
    Z_IMAGE_TURBO = "Z-Image-Turbo"


class ModelLoader(ForgeModel):
    """z_image_turbo model loader."""

    _VARIANTS = {
        ModelVariant.Z_IMAGE_TURBO_VAE: ModelConfig(
            pretrained_model_name=VAE_REPO_ID,
        ),
        ModelVariant.Z_IMAGE_TURBO: ModelConfig(
            pretrained_model_name=PIPELINE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Z_IMAGE_TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_IMAGE_TURBO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKL:
        """Load VAE from single-file safetensors."""
        vae_path = hf_hub_download(
            repo_id=VAE_REPO_ID,
            filename="split_files/vae/ae.safetensors",
        )

        config_dir = os.path.join(os.path.dirname(__file__), "vae_config")
        self._vae = AutoencoderKL.from_single_file(
            vae_path,
            config=config_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        self._vae.eval()
        return self._vae

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the full Z-Image-Turbo pipeline."""
        self._pipe = ZImagePipeline.from_pretrained(
            PIPELINE_REPO_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the model for the selected variant.

        For Z_IMAGE_TURBO_VAE: returns AutoencoderKL instance.
        For Z_IMAGE_TURBO: returns the DiT transformer from the pipeline.
        """
        if self._variant == ModelVariant.Z_IMAGE_TURBO_VAE:
            dtype = dtype_override if dtype_override is not None else torch.float32
            if self._vae is None:
                return self._load_vae(dtype)
            if dtype_override is not None:
                self._vae = self._vae.to(dtype=dtype_override)
            return self._vae

        # Z_IMAGE_TURBO variant - return the transformer
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the selected variant.

        For Z_IMAGE_TURBO_VAE: pass vae_type="decoder" or vae_type="encoder".
        For Z_IMAGE_TURBO: returns transformer inputs (latents, timestep, prompt_embeds).
        """
        if self._variant == ModelVariant.Z_IMAGE_TURBO_VAE:
            return self._load_vae_inputs(**kwargs)
        return self._load_pipeline_inputs(**kwargs)

    def _load_vae_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the VAE variant."""
        dtype = kwargs.get("dtype_override", torch.float32)
        vae_type = kwargs.get("vae_type", "decoder")

        if vae_type == "decoder":
            return torch.randn(
                1,
                LATENT_CHANNELS,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            )
        elif vae_type == "encoder":
            return torch.randn(1, 3, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
            )

    def _load_pipeline_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the full pipeline transformer variant."""
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "A photo of an astronaut riding a horse on mars"

        if self._pipe is None:
            self._load_pipeline(dtype)

        # Encode the prompt
        prompt_embeds, _ = self._pipe.encode_prompt(
            prompt=prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        # Prepare latents
        num_channels_latents = self._pipe.transformer.in_channels
        vae_scale = self._pipe.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        # Prepare timestep (normalized: 0 = fully denoised, 1 = fully noised)
        timestep = torch.tensor([0.5], dtype=dtype)

        # The transformer expects:
        #   x: list of tensors [1, channels, 1, H, W] (one per batch)
        #   t: timestep tensor
        #   cap_feats: prompt_embeds (list of text embeddings)
        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
