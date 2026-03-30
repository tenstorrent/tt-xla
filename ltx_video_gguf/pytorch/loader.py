# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-Video GGUF model loader implementation for text-to-video generation.

This loader uses GGUF-quantized variants of the LTX-Video model from
city96/LTX-Video-gguf. The GGUF transformer is loaded via diffusers'
LTXVideoTransformer3DModel.from_single_file and plugged into an LTXPipeline
built from the original Lightricks/LTX-Video repository.

Available variants:
- Q4_0: 4-bit quantization (default)
- Q8_0: 8-bit quantization
"""

from typing import Optional

import torch
from diffusers import LTXPipeline, LTXVideoTransformer3DModel

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

GGUF_REPO = "city96/LTX-Video-gguf"
BASE_REPO = "Lightricks/LTX-Video"


class ModelVariant(StrEnum):
    """Available LTX-Video GGUF quantization variants."""

    Q4_0 = "Q4_0"
    Q8_0 = "Q8_0"


# Map variants to their GGUF filenames
_GGUF_FILES = {
    ModelVariant.Q4_0: "ltx-video-2b-v0.9-Q4_0.gguf",
    ModelVariant.Q8_0: "ltx-video-2b-v0.9-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """LTX-Video GGUF model loader for text-to-video generation."""

    _VARIANTS = {
        ModelVariant.Q4_0: ModelConfig(pretrained_model_name=GGUF_REPO),
        ModelVariant.Q8_0: ModelConfig(pretrained_model_name=GGUF_REPO),
    }

    DEFAULT_VARIANT = ModelVariant.Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX-Video GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16):
        """Load the LTXPipeline with a GGUF-quantized transformer."""
        gguf_file = _GGUF_FILES[self._variant]

        transformer = LTXVideoTransformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
            torch_dtype=dtype,
        )

        self.pipe = LTXPipeline.from_pretrained(
            BASE_REPO,
            transformer=transformer,
            torch_dtype=dtype,
        )

        self.pipe.enable_attention_slicing()

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized LTX-Video transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipe is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype=dtype_override)
        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the LTX-Video transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipe is None:
            self._load_pipeline(dtype)

        prompt = "A woman with long brown hair and light skin smiles at another woman"
        height = 128
        width = 128
        num_frames = 9
        num_images_per_prompt = 1

        # Compute latent dimensions
        vae_spatial = self.pipe.vae_scale_factor_spatial
        vae_temporal = self.pipe.vae_scale_factor_temporal

        latent_height = height // vae_spatial
        latent_width = width // vae_spatial
        latent_num_frames = (num_frames - 1) // vae_temporal + 1

        in_channels = self.pipe.transformer.config.in_channels
        video_seq_len = latent_num_frames * latent_height * latent_width

        # Text encoding
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoder_hidden_states = self.pipe.text_encoder(text_inputs.input_ids,)[
            0
        ].to(dtype=dtype)
        encoder_hidden_states = encoder_hidden_states.repeat(
            batch_size * num_images_per_prompt, 1, 1
        )

        encoder_attention_mask = text_inputs.attention_mask.to(dtype=dtype)
        encoder_attention_mask = encoder_attention_mask.repeat(
            batch_size * num_images_per_prompt, 1
        )

        # Latents
        hidden_states = torch.randn(
            batch_size * num_images_per_prompt,
            video_seq_len,
            in_channels,
            dtype=dtype,
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(
            batch_size * num_images_per_prompt
        )

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
            "num_frames": latent_num_frames,
            "height": latent_height,
            "width": latent_width,
            "return_dict": False,
        }
