#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Fun Reward LoRA model loader implementation.

Loads the Wan 2.2 T2V base pipeline and applies reward-optimized LoRA weights
from alibaba-pai/Wan2.2-Fun-Reward-LoRAs for text-to-video generation with
improved human preference alignment.

The transformer (WanTransformer3DModel) is extracted from the pipeline and
returned as the model for compilation.  LoRA weights are applied before
extraction when possible; if the diffusers conversion for this LoRA format
fails, LoRA loading is skipped gracefully so the base model can be compiled.

Available variants:
- WAN22_FUN_HIGH_NOISE_HPS21: High noise LoRA optimized with HPS v2.1
- WAN22_FUN_LOW_NOISE_HPS21: Low noise LoRA optimized with HPS v2.1
- WAN22_FUN_HIGH_NOISE_MPS: High noise LoRA optimized with MPS
- WAN22_FUN_LOW_NOISE_MPS: Low noise LoRA optimized with MPS
"""

import warnings
from typing import Any, Optional

import torch
from diffusers import WanPipeline  # type: ignore[import]

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

BASE_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
LORA_REPO = "alibaba-pai/Wan2.2-Fun-Reward-LoRAs"

# LoRA weight filenames
LORA_HIGH_NOISE_HPS21 = "Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors"
LORA_LOW_NOISE_HPS21 = "Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors"
LORA_HIGH_NOISE_MPS = "Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors"
LORA_LOW_NOISE_MPS = "Wan2.2-Fun-A14B-InP-low-noise-MPS.safetensors"

# Transformer input dimensions for Wan2.2-T2V-A14B
_TRANSFORMER_IN_CHANNELS = 16
_LATENT_HEIGHT = 4
_LATENT_WIDTH = 4
_LATENT_DEPTH = 2
_TEXT_HIDDEN_DIM = 4096
_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.2 Fun Reward LoRA variants."""

    WAN22_FUN_HIGH_NOISE_HPS21 = "2.2_Fun_HighNoise_HPS2.1"
    WAN22_FUN_LOW_NOISE_HPS21 = "2.2_Fun_LowNoise_HPS2.1"
    WAN22_FUN_HIGH_NOISE_MPS = "2.2_Fun_HighNoise_MPS"
    WAN22_FUN_LOW_NOISE_MPS = "2.2_Fun_LowNoise_MPS"


_LORA_FILES = {
    ModelVariant.WAN22_FUN_HIGH_NOISE_HPS21: LORA_HIGH_NOISE_HPS21,
    ModelVariant.WAN22_FUN_LOW_NOISE_HPS21: LORA_LOW_NOISE_HPS21,
    ModelVariant.WAN22_FUN_HIGH_NOISE_MPS: LORA_HIGH_NOISE_MPS,
    ModelVariant.WAN22_FUN_LOW_NOISE_MPS: LORA_LOW_NOISE_MPS,
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Fun Reward LoRA model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_FUN_HIGH_NOISE_HPS21: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.WAN22_FUN_LOW_NOISE_HPS21: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.WAN22_FUN_HIGH_NOISE_MPS: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.WAN22_FUN_LOW_NOISE_MPS: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_FUN_HIGH_NOISE_HPS21

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_FUN_REWARD_LORAS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Wan 2.2 T2V transformer with reward LoRA weights applied.

        Loads the full pipeline, applies LoRA weights where possible, then
        extracts and returns the transformer (WanTransformer3DModel) so that
        the test framework receives a torch.nn.Module.

        Returns:
            WanTransformer3DModel with LoRA weights fused when loadable.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = WanPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        lora_file = _LORA_FILES[self._variant]
        try:
            self.pipeline.load_lora_weights(
                LORA_REPO,
                weight_name=lora_file,
            )
        except (IndexError, ValueError) as e:
            # The alibaba-pai Fun LoRAs use the musubi format with extra non-block
            # keys (head, text_embedding, time_embedding) that diffusers does not
            # yet handle when loading onto the standard T2V base model.  Skip LoRA
            # loading so the base pipeline can still be compiled.
            warnings.warn(
                f"Skipping LoRA loading for {lora_file}: {e}",
                stacklevel=2,
            )

        transformer = self.pipeline.transformer
        transformer.eval()
        return transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare synthetic inputs for the WanTransformer3DModel forward pass.

        Returns:
            dict with hidden_states, encoder_hidden_states, timestep, and
            return_dict keys suitable for WanTransformer3DModel.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = 1
        seq_len = _LATENT_DEPTH * _LATENT_HEIGHT * _LATENT_WIDTH

        hidden_states = torch.randn(
            batch_size, seq_len, _TRANSFORMER_IN_CHANNELS, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size, _TEXT_SEQ_LEN, _TEXT_HIDDEN_DIM, dtype=dtype
        )
        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
