# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Video Diffusion Img2Vid-XT 1.1 model loader for tt_forge_models.

Stable Video Diffusion is an image-to-video diffusion model by Stability AI
that generates 25-frame video clips at 1024x576 resolution from a single
conditioning image.

Repository: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1

Available subfolders:
- unet: The UNet3D conditional denoiser
- vae: Temporal video autoencoder (encoder/decoder)
- image_encoder: CLIP vision encoder for image conditioning
"""

from typing import Any, Optional

import torch
from diffusers import StableVideoDiffusionPipeline

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

SUPPORTED_SUBFOLDERS = {"unet", "vae", "image_encoder"}


class ModelVariant(StrEnum):
    """Available Stable Video Diffusion variants."""

    IMG2VID_XT_1_1 = "img2vid-xt-1.1"


class ModelLoader(ForgeModel):
    """
    Loader for Stable Video Diffusion Img2Vid-XT 1.1 model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'unet': The UNet3D conditional denoiser
    - 'vae': Temporal video autoencoder (AutoencoderKLTemporalDecoder)
    - 'image_encoder': CLIP vision encoder for image conditioning
    """

    _VARIANTS = {
        ModelVariant.IMG2VID_XT_1_1: ModelConfig(
            pretrained_model_name="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IMG2VID_XT_1_1

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        subfolder: Optional[str] = None,
    ):
        super().__init__(variant)
        if subfolder is not None and subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder
        self.pipeline: Optional[StableVideoDiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Stable Video Diffusion",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self, dtype: torch.dtype, **kwargs
    ) -> StableVideoDiffusionPipeline:
        model_kwargs = {"torch_dtype": dtype}
        model_kwargs |= kwargs
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float16

        if self.pipeline is None:
            self._load_pipeline(dtype, **kwargs)

        if self._subfolder == "vae":
            return self.pipeline.vae
        elif self._subfolder == "image_encoder":
            return self.pipeline.image_encoder
        elif self._subfolder == "unet" or self._subfolder is None:
            return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "unet" or self._subfolder is None:
            return self._load_unet_inputs(dtype)
        elif self._subfolder == "vae":
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return self._load_vae_decoder_inputs(dtype)
            else:
                return self._load_vae_encoder_inputs(dtype)
        elif self._subfolder == "image_encoder":
            return self._load_image_encoder_inputs(dtype)

    def _load_unet_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the UNet3D conditional denoiser."""
        batch_size = 1
        config = self.pipeline.unet.config

        # Use small latent dimensions for testing
        num_frames = 2
        latent_height = 2
        latent_width = 2

        in_channels = config.in_channels
        sample = torch.randn(
            batch_size,
            num_frames,
            in_channels,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        timestep = torch.tensor([0.5], dtype=dtype)

        # Image conditioning via CLIP embeddings
        encoder_hidden_states = torch.randn(
            batch_size,
            1,
            config.cross_attention_dim,
            dtype=dtype,
        )

        # Additional conditioning (fps, motion bucket, noise augmentation)
        added_time_ids = torch.tensor([[6.0, 127.0, 0.0]], dtype=dtype)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "added_time_ids": added_time_ids,
            "return_dict": False,
        }

    def _load_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic decoder inputs for the temporal video VAE."""
        latent_channels = self.pipeline.vae.config.latent_channels
        return {
            "sample": torch.randn(1, latent_channels, 2, 2, 2, dtype=dtype),
            "num_frames": 2,
        }

    def _load_vae_encoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic encoder inputs for the temporal video VAE."""
        return {
            "sample": torch.randn(1, 3, 2, 64, 64, dtype=dtype),
        }

    def _load_image_encoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the CLIP image encoder."""
        return {
            "pixel_values": torch.randn(1, 3, 224, 224, dtype=dtype),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        if hasattr(output, "image_embeds"):
            return output.image_embeds
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output
