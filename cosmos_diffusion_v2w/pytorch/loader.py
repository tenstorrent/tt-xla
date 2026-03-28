# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos 1.0 Diffusion 14B Video2World model loader for tt_forge_models.

Cosmos Video2World is a 14B parameter diffusion-based world foundation model by NVIDIA
that generates physics-aware future video frames from an input video (or image) plus a
text prompt.

Repository: https://huggingface.co/nvidia/Cosmos-1.0-Diffusion-14B-Video2World

Available subfolders:
- transformer: The diffusion transformer (DiT) denoiser
- vae: Video tokenizer (encoder/decoder)
- text_encoder: Text encoder for prompt conditioning
"""

from typing import Any, Optional

import torch
from diffusers import CosmosVideoToWorldPipeline

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

SUPPORTED_SUBFOLDERS = {"transformer", "vae", "text_encoder"}


class ModelVariant(StrEnum):
    """Available Cosmos Diffusion Video2World variants."""

    V1_14B = "1.0-14B"


class ModelLoader(ForgeModel):
    """
    Loader for Cosmos 1.0 Diffusion 14B Video2World model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': The diffusion transformer denoiser
    - 'vae': Video tokenizer (AutoencoderKL)
    - 'text_encoder': T5-based text encoder
    """

    _VARIANTS = {
        ModelVariant.V1_14B: ModelConfig(
            pretrained_model_name="nvidia/Cosmos-1.0-Diffusion-14B-Video2World",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_14B

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
        self.pipeline: Optional[CosmosVideoToWorldPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Cosmos Diffusion Video2World",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self, dtype: torch.dtype, **kwargs
    ) -> CosmosVideoToWorldPipeline:
        model_kwargs = {"torch_dtype": dtype}
        model_kwargs |= kwargs
        self.pipeline = CosmosVideoToWorldPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype, **kwargs)

        if self._subfolder == "vae":
            return self.pipeline.vae
        elif self._subfolder == "text_encoder":
            return self.pipeline.text_encoder
        elif self._subfolder == "transformer" or self._subfolder is None:
            return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "transformer" or self._subfolder is None:
            return self._load_transformer_inputs(dtype)
        elif self._subfolder == "vae":
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return self._load_vae_decoder_inputs(dtype)
            else:
                return self._load_vae_encoder_inputs(dtype)
        elif self._subfolder == "text_encoder":
            return self._load_text_encoder_inputs(dtype)

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the Cosmos diffusion transformer."""
        batch_size = 1
        config = self.pipeline.transformer.config

        # Use small latent dimensions for testing
        num_latent_frames = 2
        latent_height = 2
        latent_width = 2

        in_channels = config.in_channels
        hidden_states = torch.randn(
            batch_size,
            in_channels,
            num_latent_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        # Text conditioning
        encoder_hidden_states = torch.randn(
            batch_size, 8, config.joint_attention_dim, dtype=dtype
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

    def _load_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic decoder inputs for the video VAE."""
        latent_channels = self.pipeline.vae.config.latent_channels
        return {
            "sample": torch.randn(1, latent_channels, 2, 2, 2, dtype=dtype),
        }

    def _load_vae_encoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic encoder inputs for the video VAE."""
        return {
            "sample": torch.randn(1, 3, 9, 64, 64, dtype=dtype),
        }

    def _load_text_encoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the text encoder."""
        return {
            "input_ids": torch.randint(0, 1000, (1, 16)),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output
