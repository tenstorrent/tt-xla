# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FramePackI2V_HY model loader for tt_forge_models.

FramePack is an image-to-video generation model based on HunyuanVideo that uses
next-frame-section prediction with context compression ("frame packing") to
generate videos progressively from a single input image and text prompt.

The model uses a 13B-parameter 3D transformer with dual-stream and single-stream
attention layers, RoPE positional encoding, and an image projection layer for
conditioning on a reference image.

Repository:
- https://huggingface.co/lllyasviel/FramePackI2V_HY

Available subfolders:
- transformer: HunyuanVideoTransformer3DModel
- vae: AutoencoderKLHunyuanVideo
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLHunyuanVideo, HunyuanVideoPipeline

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

HUNYUAN_BASE_REPO = "hunyuanvideo-community/HunyuanVideo"
FRAMEPACK_REPO = "lllyasviel/FramePackI2V_HY"

SUPPORTED_SUBFOLDERS = {"transformer", "vae"}


class ModelVariant(StrEnum):
    """Available FramePackI2V_HY variants."""

    DEFAULT = "default"


class ModelLoader(ForgeModel):
    """
    Loader for FramePackI2V_HY image-to-video generation model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': HunyuanVideoTransformer3DModel
    - 'vae': AutoencoderKLHunyuanVideo
    """

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name=FRAMEPACK_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

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
            model="FramePackI2V_HY",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> HunyuanVideoPipeline:
        self.pipeline = HunyuanVideoPipeline.from_pretrained(
            HUNYUAN_BASE_REPO,
            torch_dtype=dtype,
        )
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "vae":
            return self.pipeline.vae
        elif self._subfolder == "transformer":
            return self.pipeline.transformer
        else:
            return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "vae":
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return self._load_vae_decoder_inputs(dtype)
            else:
                return self._load_vae_encoder_inputs(dtype)
        else:
            return self._load_transformer_inputs(dtype)

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the HunyuanVideo transformer forward pass."""
        batch_size = 1
        config = self.pipeline.transformer.config

        latent_num_frames = 2
        latent_height = 2
        latent_width = 2
        video_seq_len = latent_num_frames * latent_height * latent_width

        hidden_states = torch.randn(
            batch_size, video_seq_len, config.in_channels, dtype=dtype
        )

        encoder_hidden_states = torch.randn(
            batch_size, 8, config.text_embed_dim, dtype=dtype
        )
        encoder_attention_mask = torch.ones(batch_size, 8, dtype=dtype)

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "timestep": timestep,
            "return_dict": False,
        }

    def _load_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic decoder inputs for the HunyuanVideo VAE."""
        latent_channels = self.pipeline.vae.config.latent_channels
        return {
            "sample": torch.randn(1, latent_channels, 2, 2, 2, dtype=dtype),
        }

    def _load_vae_encoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic encoder inputs for the HunyuanVideo VAE."""
        return {
            "sample": torch.randn(1, 3, 9, 64, 64, dtype=dtype),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
