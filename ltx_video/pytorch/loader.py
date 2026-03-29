# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-Video model loader for tt_forge_models.

LTX-Video is a text-to-video diffusion model that uses a DiT (Diffusion Transformer)
architecture to generate video from text prompts.

Repository: https://huggingface.co/Isi99999/LTX-Video

Available subfolders:
- transformer: LTXVideoTransformer3DModel
- vae: AutoencoderKLLTXVideo
- text_encoder: T5 text encoder for prompt conditioning
"""

from typing import Any, Optional

import torch
from diffusers import LTXPipeline

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
    """Available LTX-Video variants."""

    DEFAULT = "default"


class ModelLoader(ForgeModel):
    """
    Loader for LTX-Video text-to-video generation model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': LTXVideoTransformer3DModel
    - 'vae': AutoencoderKLLTXVideo
    - 'text_encoder': T5 text encoder for prompt conditioning
    """

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="Isi99999/LTX-Video",
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
        self.pipeline: Optional[LTXPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LTX-Video",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype, **kwargs) -> LTXPipeline:
        model_kwargs = {"torch_dtype": dtype}
        model_kwargs |= kwargs
        self.pipeline = LTXPipeline.from_pretrained(
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
        """Prepare synthetic inputs for the LTX-Video transformer."""
        batch_size = 1
        config = self.pipeline.transformer.config

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

        encoder_hidden_states = torch.randn(
            batch_size, 8, config.cross_attention_dim, dtype=dtype
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
