# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX model loader for tt_forge_models.

LTX-2 is a 19B parameter and LTX-2.3 is a 22B parameter DiT (Diffusion Transformer)
audio-video foundation model by Lightricks. They generate synchronized video and audio
within a single model.

Repositories:
- https://huggingface.co/Lightricks/LTX-2
- https://huggingface.co/rootonchair/LTX-2-19b-distilled
- https://huggingface.co/Lightricks/LTX-2.3
- https://huggingface.co/Lightricks/LTX-2.3-nvfp4

Available subfolders:
- transformer: LTX2VideoTransformer3DModel
- vae: AutoencoderKLLTX2Video
- audio_vae: AutoencoderKLLTX2Audio
"""

from typing import Any, Optional

import torch
from diffusers import LTX2Pipeline
from diffusers.models import LTX2VideoTransformer3DModel

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

SUPPORTED_SUBFOLDERS = {"transformer", "vae", "audio_vae"}

FP8_CHECKPOINT_URL = "https://huggingface.co/Lightricks/LTX-2.3-fp8/blob/main/ltx-2.3-22b-distilled-fp8.safetensors"
NVFP4_CHECKPOINT_URL = "https://huggingface.co/Lightricks/LTX-2.3-nvfp4/blob/main/ltx-2.3-22b-dev-nvfp4.safetensors"


class ModelVariant(StrEnum):
    """Available LTX variants."""

    LTX_2 = "2"
    LTX_2_DISTILLED = "2-distilled"
    LTX_2_3 = "2.3"
    LTX_2_3_FP8 = "2.3-fp8"
    LTX_2_3_NVFP4 = "2.3-nvfp4"


class ModelLoader(ForgeModel):
    """
    Loader for LTX-2.3 audio-video generation model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': LTX2VideoTransformer3DModel (~22B params)
    - 'vae': AutoencoderKLLTX2Video
    - 'audio_vae': AutoencoderKLLTX2Audio

    The FP8 variant loads the transformer directly from a single safetensors
    checkpoint (requires diffusers>=0.38.0.dev0 from GitHub main).
    """

    _VARIANTS = {
        ModelVariant.LTX_2: ModelConfig(
            pretrained_model_name="Lightricks/LTX-2",
        ),
        ModelVariant.LTX_2_DISTILLED: ModelConfig(
            pretrained_model_name="rootonchair/LTX-2-19b-distilled",
        ),
        ModelVariant.LTX_2_3: ModelConfig(
            pretrained_model_name="Lightricks/LTX-2.3",
        ),
        ModelVariant.LTX_2_3_FP8: ModelConfig(
            pretrained_model_name="Lightricks/LTX-2.3-fp8",
        ),
        ModelVariant.LTX_2_3_NVFP4: ModelConfig(
            pretrained_model_name="Lightricks/LTX-2.3-nvfp4",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LTX_2_3

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
        self.pipeline: Optional[LTX2Pipeline] = None
        self._transformer: Optional[LTX2VideoTransformer3DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LTX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_fp8_transformer(self, dtype: torch.dtype) -> LTX2VideoTransformer3DModel:
        """Load the FP8 transformer directly from a single safetensors file."""
        self._transformer = LTX2VideoTransformer3DModel.from_single_file(
            FP8_CHECKPOINT_URL,
            torch_dtype=dtype,
            cross_attn_mod=True,
            audio_cross_attn_mod=True,
            low_cpu_mem_usage=False,
        )
        return self._transformer

    def _load_nvfp4_transformer(
        self, dtype: torch.dtype
    ) -> LTX2VideoTransformer3DModel:
        """Load the NVFP4 transformer directly from a single safetensors file."""
        self._transformer = LTX2VideoTransformer3DModel.from_single_file(
            NVFP4_CHECKPOINT_URL,
            torch_dtype=dtype,
            cross_attn_mod=True,
            audio_cross_attn_mod=True,
            low_cpu_mem_usage=False,
        )
        return self._transformer

    def _load_pipeline(self, dtype: torch.dtype) -> LTX2Pipeline:
        self.pipeline = LTX2Pipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._variant == ModelVariant.LTX_2_3_FP8:
            if self._transformer is None:
                self._load_fp8_transformer(dtype)
            return self._transformer

        if self._variant == ModelVariant.LTX_2_3_NVFP4:
            if self._transformer is None:
                self._load_nvfp4_transformer(dtype)
            return self._transformer

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "vae":
            return self.pipeline.vae
        elif self._subfolder == "audio_vae":
            return self.pipeline.audio_vae
        elif self._subfolder == "transformer" or self._subfolder is None:
            return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._variant == ModelVariant.LTX_2_3_FP8:
            if self._transformer is None:
                self._load_fp8_transformer(dtype)
            return self._load_fp8_transformer_inputs(dtype)

        if self._variant == ModelVariant.LTX_2_3_NVFP4:
            if self._transformer is None:
                self._load_nvfp4_transformer(dtype)
            return self._load_fp8_transformer_inputs(dtype)

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
        elif self._subfolder == "audio_vae":
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return self._load_audio_vae_decoder_inputs(dtype)
            else:
                return self._load_audio_vae_encoder_inputs(dtype)

    def _load_fp8_transformer_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the FP8 LTX2 transformer forward pass."""
        batch_size = 1
        config = self._transformer.config

        latent_num_frames = 2
        latent_height = 2
        latent_width = 2
        video_seq_len = latent_num_frames * latent_height * latent_width
        frame_rate = 24.0

        hidden_states = torch.randn(
            batch_size, video_seq_len, config.in_channels, dtype=dtype
        )
        audio_hidden_states = torch.randn(
            batch_size, 2, config.audio_in_channels, dtype=dtype
        )

        caption_channels = config.caption_channels
        encoder_hidden_states = torch.randn(
            batch_size, 8, caption_channels, dtype=dtype
        )
        audio_encoder_hidden_states = torch.randn(
            batch_size, 8, caption_channels, dtype=dtype
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)
        audio_timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)
        sigma = torch.tensor([0.5], dtype=dtype).expand(batch_size)
        audio_sigma = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_encoder_hidden_states": audio_encoder_hidden_states,
            "timestep": timestep,
            "audio_timestep": audio_timestep,
            "sigma": sigma,
            "audio_sigma": audio_sigma,
            "num_frames": latent_num_frames,
            "height": latent_height,
            "width": latent_width,
            "fps": frame_rate,
            "audio_num_frames": 2,
            "return_dict": False,
        }

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the LTX2 transformer forward pass."""
        batch_size = 1
        # Use small dimensions for testing
        height = 64
        width = 64
        num_frames = 9  # Must be 8k+1 format

        # Compute latent dimensions using pipeline compression ratios
        vae_spatial = self.pipeline.vae_spatial_compression_ratio  # 32
        vae_temporal = self.pipeline.vae_temporal_compression_ratio  # 8

        latent_height = height // vae_spatial
        latent_width = width // vae_spatial
        latent_num_frames = (num_frames - 1) // vae_temporal + 1

        # Video latents: packed shape [B, F*H*W, C]
        in_channels = self.pipeline.transformer.config.in_channels  # 128
        video_seq_len = latent_num_frames * latent_height * latent_width
        hidden_states = torch.randn(batch_size, video_seq_len, in_channels, dtype=dtype)

        # Audio latents: packed shape [B, audio_seq_len, audio_C * mel_bins]
        frame_rate = 24.0
        duration_s = num_frames / frame_rate
        audio_sampling_rate = self.pipeline.audio_sampling_rate
        audio_hop_length = self.pipeline.audio_hop_length
        audio_vae_temporal = self.pipeline.audio_vae_temporal_compression_ratio
        audio_latents_per_second = (
            audio_sampling_rate / audio_hop_length / float(audio_vae_temporal)
        )
        audio_num_frames = max(1, round(duration_s * audio_latents_per_second))
        num_mel_bins = (
            self.pipeline.audio_vae.config.mel_bins
            if getattr(self.pipeline, "audio_vae", None) is not None
            else 64
        )
        audio_vae_mel = self.pipeline.audio_vae_mel_compression_ratio
        latent_mel_bins = num_mel_bins // audio_vae_mel
        audio_in_channels = self.pipeline.transformer.config.audio_in_channels  # 128
        # Audio is packed as [B, L, C*M]
        audio_hidden_states = torch.randn(
            batch_size,
            audio_num_frames,
            audio_in_channels * latent_mel_bins,
            dtype=dtype,
        )

        # Text embeddings from connectors
        max_seq_len = 64
        caption_channels = self.pipeline.transformer.config.caption_channels
        encoder_hidden_states = torch.randn(
            batch_size, max_seq_len, caption_channels, dtype=dtype
        )
        audio_encoder_hidden_states = torch.randn(
            batch_size, max_seq_len, caption_channels, dtype=dtype
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_encoder_hidden_states": audio_encoder_hidden_states,
            "timestep": timestep,
            "num_frames": latent_num_frames,
            "height": latent_height,
            "width": latent_width,
            "fps": frame_rate,
            "audio_num_frames": audio_num_frames,
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

    def _load_audio_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic decoder inputs for the audio VAE."""
        latent_channels = self.pipeline.audio_vae.config.latent_channels
        mel_bins = self.pipeline.audio_vae.config.mel_bins
        audio_vae_mel = self.pipeline.audio_vae_mel_compression_ratio
        latent_mel_bins = mel_bins // audio_vae_mel
        return {
            "sample": torch.randn(1, latent_channels, 4, latent_mel_bins, dtype=dtype),
        }

    def _load_audio_vae_encoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic encoder inputs for the audio VAE."""
        mel_bins = self.pipeline.audio_vae.config.mel_bins
        return {
            "sample": torch.randn(1, 1, 16, mel_bins, dtype=dtype),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
