# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-2509-Fusion LoRA model loader implementation.

Loads the Qwen-Image-Edit-2509 diffusion transformer and applies
the dx8152/Qwen-Image-Edit-2509-Fusion LoRA adapter for image fusion
and product perspective/lighting correction.

Available variants:
- QWEN_IMAGE_EDIT_2509_FUSION: Image fusion LoRA (bf16)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageTransformer2DModel
from peft import PeftModel

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "dx8152/Qwen-Image-Edit-2509-Fusion"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-2509-Fusion model variants."""

    QWEN_IMAGE_EDIT_2509_FUSION = "Edit_2509_Fusion"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-2509-Fusion LoRA model loader for image fusion."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_2509_FUSION: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_2509_FUSION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_2509_FUSION",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the Qwen-Image-Edit transformer with fusion LoRA weights applied.

        Returns:
            QwenImageTransformer2DModel with LoRA adapter merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        transformer = QwenImageTransformer2DModel.from_pretrained(
            BASE_MODEL,
            subfolder="transformer",
            torch_dtype=dtype,
        )

        transformer = PeftModel.from_pretrained(transformer, LORA_REPO)
        transformer = transformer.merge_and_unload()
        transformer.eval()

        self._transformer = transformer
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        # From model config: in_channels=64 (img_in linear input dimension)
        img_dim = 64
        # joint_attention_dim from config = 3584
        text_dim = 3584
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
