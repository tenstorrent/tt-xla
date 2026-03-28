# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Image-Edit model loader implementation for image editing.

Uses the LongCatImageEditPipeline from diffusers with the
LongCatImageTransformer2DModel diffusion transformer.
"""
from typing import Any, Optional

import torch
from diffusers import LongCatImageEditPipeline

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


class ModelVariant(StrEnum):
    """Available LongCat-Image-Edit model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """LongCat-Image-Edit model loader for bilingual image editing tasks."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="meituan-longcat/LongCat-Image-Edit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LongCat-Image-Edit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        pipe_kwargs = {}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = LongCatImageEditPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )
        self.pipe.enable_model_cpu_offload()

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        return self.pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the LongCatImageTransformer2DModel.

        Returns a dict matching the transformer's forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        # From transformer config: in_channels=64
        img_dim = 64
        # joint_attention_dim=3584
        text_dim = 3584
        # pooled_projection_dim=3584
        pooled_dim = 3584

        txt_seq_len = 32
        # img_seq_len for a small latent spatial size
        height, width = 8, 8
        img_seq_len = height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        pooled_projections = torch.randn(batch_size, pooled_dim, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        txt_ids = torch.zeros(txt_seq_len, 3, dtype=dtype)
        img_ids = torch.zeros(img_seq_len, 3, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
        }
