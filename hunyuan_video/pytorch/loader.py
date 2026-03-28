# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanVideo model loader for tt_forge_models.

HunyuanVideo 1.5 is an 8.3B parameter DiT (Diffusion Transformer) video generation
model by Tencent. It supports text-to-video generation with a 3D Causal VAE for
spatial (16x) and temporal (4x) compression.

Repository:
- https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v

Available subfolders:
- transformer: HunyuanVideoTransformer3DModel (~8.3B params)
- vae: AutoencoderKLHunyuanVideo
"""

from typing import Any, Optional

import torch
from diffusers import HunyuanVideoPipeline

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
    """Available HunyuanVideo variants."""

    HUNYUAN_VIDEO_720P = "720p"


class ModelLoader(ForgeModel):
    """
    Loader for HunyuanVideo 1.5 video generation model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': HunyuanVideoTransformer3DModel (~8.3B params)
    - 'vae': AutoencoderKLHunyuanVideo
    """

    _VARIANTS = {
        ModelVariant.HUNYUAN_VIDEO_720P: ModelConfig(
            pretrained_model_name="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUNYUAN_VIDEO_720P

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
        self.pipeline: Optional[HunyuanVideoPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HunyuanVideo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> HunyuanVideoPipeline:
        self.pipeline = HunyuanVideoPipeline.from_pretrained(
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
        """Prepare synthetic inputs for the HunyuanVideo transformer forward pass."""
        batch_size = 1
        config = self.pipeline.transformer.config

        # Use small dimensions for testing
        height = 64
        width = 64
        num_frames = 9

        # Compute latent dimensions using VAE compression ratios
        vae_spatial = self.pipeline.vae_scale_factor_spatial  # 8
        vae_temporal = self.pipeline.vae_scale_factor_temporal  # 4

        latent_height = height // vae_spatial
        latent_width = width // vae_spatial
        latent_num_frames = (num_frames - 1) // vae_temporal + 1

        in_channels = config.in_channels
        hidden_states = torch.randn(
            batch_size,
            latent_num_frames,
            in_channels,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        # Text encoder hidden states
        text_seq_len = 64
        text_embed_dim = config.text_embed_dim
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, text_embed_dim, dtype=dtype
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        guidance = torch.tensor([6.0], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "guidance": guidance,
            "return_dict": False,
        }

    def _load_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic decoder inputs for the video VAE."""
        latent_channels = self.pipeline.vae.config.latent_channels
        return {
            "sample": torch.randn(1, latent_channels, 3, 8, 8, dtype=dtype),
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
