# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ModelScope text-to-video model loader for tt_forge_models.

ModelScope Text-to-Video MS 1.7B is a diffusion-based text-to-video generation
model with ~1.7B parameters. It generates videos from English text descriptions
using a UNet3D diffusion architecture.

Repository:
- https://huggingface.co/ali-vilab/text-to-video-ms-1.7b

Available subfolders:
- unet: UNet3DConditionModel
- vae: AutoencoderKL
- text_encoder: CLIPTextModel
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

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

SUPPORTED_SUBFOLDERS = {"unet", "vae", "text_encoder"}


class ModelVariant(StrEnum):
    """Available text-to-video-ms variants."""

    TEXT_TO_VIDEO_MS_1_7B = "1.7b"


class ModelLoader(ForgeModel):
    """
    Loader for ModelScope Text-to-Video MS 1.7B.

    Supports loading the full pipeline or individual components via subfolder:
    - 'unet': UNet3DConditionModel (~1.7B params)
    - 'vae': AutoencoderKL
    - 'text_encoder': CLIPTextModel
    """

    _VARIANTS = {
        ModelVariant.TEXT_TO_VIDEO_MS_1_7B: ModelConfig(
            pretrained_model_name="ali-vilab/text-to-video-ms-1.7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEXT_TO_VIDEO_MS_1_7B

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
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Text-to-Video-MS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> DiffusionPipeline:
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "vae":
            return self.pipeline.vae
        elif self._subfolder == "text_encoder":
            return self.pipeline.text_encoder
        elif self._subfolder == "unet" or self._subfolder is None:
            return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32

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
        elif self._subfolder == "text_encoder":
            return self._load_text_encoder_inputs()

    def _load_unet_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the UNet3D forward pass."""
        batch_size = 1
        unet = self.pipeline.unet
        config = unet.config

        # Use small dimensions for testing
        num_frames = 2
        height = 2
        width = 2

        sample = torch.randn(
            batch_size,
            config.in_channels,
            num_frames,
            height,
            width,
            dtype=dtype,
        )

        timestep = torch.tensor([0.5], dtype=dtype)

        encoder_hidden_states = torch.randn(
            batch_size, 8, config.cross_attention_dim, dtype=dtype
        )

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
        }

    def _load_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic decoder inputs for the VAE."""
        latent_channels = self.pipeline.vae.config.latent_channels
        return {
            "sample": torch.randn(1, latent_channels, 2, 2, 2, dtype=dtype),
        }

    def _load_vae_encoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic encoder inputs for the VAE."""
        return {
            "sample": torch.randn(1, 3, 2, 64, 64, dtype=dtype),
        }

    def _load_text_encoder_inputs(self) -> dict:
        """Prepare synthetic inputs for the text encoder."""
        return {
            "input_ids": torch.randint(0, 1000, (1, 8)),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output
