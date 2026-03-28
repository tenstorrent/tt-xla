# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CogVideoX model loader for tt_forge_models.

CogVideoX-5B-I2V is a 5B parameter image-to-video diffusion transformer model.
It generates 6-second videos (49 frames at 8 FPS) at 720x480 resolution from
a reference image and text prompt.

Repository: https://huggingface.co/zai-org/CogVideoX-5b-I2V

Available subfolders:
- transformer: CogVideoXTransformer3DModel
- vae: AutoencoderKLCogVideoX
- text_encoder: T5EncoderModel
"""

from typing import Any, Optional

import torch
from diffusers import CogVideoXImageToVideoPipeline

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
    """Available CogVideoX variants."""

    COGVIDEOX_5B_I2V = "5b-I2V"


class ModelLoader(ForgeModel):
    """
    Loader for CogVideoX image-to-video generation model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': CogVideoXTransformer3DModel (~5B params)
    - 'vae': AutoencoderKLCogVideoX
    - 'text_encoder': T5EncoderModel
    """

    _VARIANTS = {
        ModelVariant.COGVIDEOX_5B_I2V: ModelConfig(
            pretrained_model_name="zai-org/CogVideoX-5b-I2V",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COGVIDEOX_5B_I2V

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
        self.pipeline: Optional[CogVideoXImageToVideoPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CogVideoX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> CogVideoXImageToVideoPipeline:
        self.pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
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
            return self._load_text_encoder_inputs()

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the CogVideoX transformer forward pass."""
        batch_size = 1
        config = self.pipeline.transformer.config

        # CogVideoX uses patched 3D latents
        # Video: 49 frames, 480x720 -> latent dims after VAE compression
        num_frames = 2
        height = 2
        width = 2

        in_channels = config.in_channels
        hidden_states = torch.randn(
            batch_size, num_frames, in_channels, height, width, dtype=dtype
        )

        # Text encoder hidden states
        encoder_hidden_states = torch.randn(
            batch_size, 226, config.text_embed_dim, dtype=dtype
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

    def _load_text_encoder_inputs(self) -> dict:
        """Prepare synthetic inputs for the T5 text encoder."""
        batch_size = 1
        seq_len = 226
        return {
            "input_ids": torch.randint(0, 32128, (batch_size, seq_len)),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output
