# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-Rapid-AIO-V4 model loader implementation.

Loads the distilled diffusion transformer from
prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V4. This is a rapid (few-step)
variant of the Qwen Image Edit family optimised for fast inference.

Available variants:
- V4: Qwen-Image-Edit-Rapid-AIO-V4 (bf16)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageTransformer2DModel

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

REPO_ID = "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V4"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-Rapid-AIO-V4 model variants."""

    V4 = "V4"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-Rapid-AIO-V4 diffusion transformer loader."""

    _VARIANTS = {
        ModelVariant.V4: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.V4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_RAPID_AIO_V4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the diffusion transformer.

        Returns:
            QwenImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            self._transformer = QwenImageTransformer2DModel.from_pretrained(
                REPO_ID,
                torch_dtype=dtype,
            )
            self._transformer.eval()
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        # From model config: in_channels=64
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
