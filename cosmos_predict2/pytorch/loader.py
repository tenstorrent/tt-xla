# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos Predict2 Video2World model loader for tt_forge_models.

Cosmos-Predict2 is a 2B parameter Diffusion Transformer (DiT) model by NVIDIA
that generates video from an input image and text prompt. It uses interleaved
self-attention, cross-attention, and feedforward layers in latent space.

Repository:
- https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World

Available subfolders:
- transformer: CosmosTransformer3DModel
- vae: AutoencoderKLWan
"""

from typing import Any, Optional

import torch
from diffusers import Cosmos2VideoToWorldPipeline

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

SUPPORTED_SUBFOLDERS = {"transformer", "vae"}


class ModelVariant(StrEnum):
    """Available Cosmos Predict2 Video2World variants."""

    V2W_2B = "2B"


class ModelLoader(ForgeModel):
    """
    Loader for NVIDIA Cosmos-Predict2 Video2World model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': CosmosTransformer3DModel (~2B params)
    - 'vae': AutoencoderKLWan
    """

    _VARIANTS = {
        ModelVariant.V2W_2B: ModelConfig(
            pretrained_model_name="nvidia/Cosmos-Predict2-2B-Video2World",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2W_2B

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
        self.pipeline: Optional[Cosmos2VideoToWorldPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="COSMOS_PREDICT2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> Cosmos2VideoToWorldPipeline:
        self.pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "vae":
            return self.pipeline.vae
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

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the Cosmos2 transformer forward pass."""
        batch_size = 1
        config = self.pipeline.transformer.config

        # Use small latent dimensions for testing
        latent_num_frames = 2
        latent_height = 2
        latent_width = 2

        in_channels = config.in_channels
        hidden_states = torch.randn(
            batch_size,
            in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        # Text encoder hidden states (text_embed_dim from config)
        text_embed_dim = config.text_embed_dim
        encoder_hidden_states = torch.randn(batch_size, 8, text_embed_dim, dtype=dtype)

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

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
