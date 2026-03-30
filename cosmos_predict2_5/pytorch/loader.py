# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos Predict2.5 14B model loader for tt_forge_models.

Cosmos Predict2.5 is a 14B parameter diffusion transformer world foundation model by
NVIDIA that generates physics-aware video from text, image, or video prompts. It
supports text-to-video, image-to-video, and video-to-video generation at 720p.

Repository: https://huggingface.co/nvidia/Cosmos-Predict2.5-14B

Available subfolders:
- transformer: The diffusion transformer (DiT) denoiser
- vae: Video tokenizer (encoder/decoder)
- text_encoder: Qwen2.5-VL text/vision encoder for prompt conditioning
"""

from typing import Any, Optional

import torch
from diffusers import Cosmos2_5_PredictBasePipeline

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
    """Available Cosmos Predict2.5 variants."""

    V2_5_14B = "2.5-14B"


class ModelLoader(ForgeModel):
    """
    Loader for Cosmos Predict2.5 14B model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': The diffusion transformer denoiser (CosmosTransformer3DModel)
    - 'vae': Video tokenizer (AutoencoderKLWan)
    - 'text_encoder': Qwen2.5-VL encoder for prompt conditioning
    """

    _VARIANTS = {
        ModelVariant.V2_5_14B: ModelConfig(
            pretrained_model_name="nvidia/Cosmos-Predict2.5-14B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_5_14B

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
        self.pipeline: Optional[Cosmos2_5_PredictBasePipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Cosmos Predict2.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self, dtype: torch.dtype, **kwargs
    ) -> Cosmos2_5_PredictBasePipeline:
        model_kwargs = {"torch_dtype": dtype}
        model_kwargs |= kwargs
        self.pipeline = Cosmos2_5_PredictBasePipeline.from_pretrained(
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
        """Prepare synthetic inputs for the Cosmos Predict2.5 diffusion transformer."""
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

        # Text conditioning - dimension depends on whether cross-attention
        # projection is used
        if config.use_crossattn_projection:
            text_dim = config.crossattn_proj_in_channels
        else:
            text_dim = config.text_embed_dim
        encoder_hidden_states = torch.randn(batch_size, 8, text_dim, dtype=dtype)

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

        # Provide padding mask if the model concatenates it
        if config.concat_padding_mask:
            inputs["padding_mask"] = torch.ones(
                batch_size, 1, latent_height, latent_width, dtype=dtype
            )

        return inputs

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
