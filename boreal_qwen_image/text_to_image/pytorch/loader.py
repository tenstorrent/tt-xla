# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Boreal-Qwen-Image model loader implementation for text-to-image generation.

Loads the Qwen/Qwen-Image diffusion transformer with LoRA adapter weights
from kudzueye/boreal-qwen-image applied and merged. Returns the transformer
component for testing.

Available variants:
- BLEND_LOW_RANK: Boreal blend style (low rank)
- GENERAL_DISCRETE_LOW_RANK: General discrete style (low rank)
- PORTRAITS_HIGH_RANK: Portrait style (high rank)
- SMALL_DISCRETE_LOW_RANK: Small discrete style (low rank)
"""

from typing import Any, Dict, Optional

import torch
from diffusers import DiffusionPipeline

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

LORA_REPO_ID = "kudzueye/boreal-qwen-image"
BASE_MODEL_ID = "Qwen/Qwen-Image"


class ModelVariant(StrEnum):
    """Available Boreal-Qwen-Image LoRA variants."""

    BLEND_LOW_RANK = "blend-low-rank"
    GENERAL_DISCRETE_LOW_RANK = "general-discrete-low-rank"
    PORTRAITS_HIGH_RANK = "portraits-high-rank"
    SMALL_DISCRETE_LOW_RANK = "small-discrete-low-rank"


_LORA_FILES = {
    ModelVariant.BLEND_LOW_RANK: "qwen-boreal-blend-low-rank.safetensors",
    ModelVariant.GENERAL_DISCRETE_LOW_RANK: "qwen-boreal-general-discrete-low-rank.safetensors",
    ModelVariant.PORTRAITS_HIGH_RANK: "qwen-boreal-portraits-portraits-high-rank.safetensors",
    ModelVariant.SMALL_DISCRETE_LOW_RANK: "qwen-boreal-small-discrete-low-rank.safetensors",
}


class ModelLoader(ForgeModel):
    """Boreal-Qwen-Image LoRA model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.BLEND_LOW_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
        ModelVariant.GENERAL_DISCRETE_LOW_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
        ModelVariant.PORTRAITS_HIGH_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
        ModelVariant.SMALL_DISCRETE_LOW_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERAL_DISCRETE_LOW_RANK

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Boreal-Qwen-Image",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image transformer with Boreal LoRA weights merged.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._transformer is None:
            pipe = DiffusionPipeline.from_pretrained(
                BASE_MODEL_ID,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            pipe.load_lora_weights(
                LORA_REPO_ID,
                weight_name=_LORA_FILES[self._variant],
            )
            pipe.fuse_lora()
            self._transformer = pipe.transformer
            self._transformer.eval()
            del pipe
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)

        return self._transformer

    def load_inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        # From Qwen-Image transformer config: in_channels=64
        img_dim = 64
        # joint_attention_dim from config = 4096
        text_dim = 4096
        txt_seq_len = 32

        # img_seq_len must equal frame * height * width for positional encoding
        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
