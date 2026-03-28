# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UFM (UniFlowMatch) model loader implementation for dense correspondence estimation
"""

import torch
from typing import Optional

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
    """Available UFM model variants."""

    REFINE = "Refine"


class ModelLoader(ForgeModel):
    """UFM model loader implementation for dense correspondence estimation."""

    _VARIANTS = {
        ModelVariant.REFINE: ModelConfig(
            pretrained_model_name="infinity1096/UFM-Refine",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.REFINE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="UFM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **kwargs
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # UFM expects two images: source and target, as (H, W, 3) tensors
        # The model's inference resolution is 560x420
        height, width = 420, 560
        source_image = torch.randint(0, 256, (height, width, 3), dtype=torch.float32)
        target_image = torch.randint(0, 256, (height, width, 3), dtype=torch.float32)

        if dtype_override is not None:
            source_image = source_image.to(dtype_override)
            target_image = target_image.to(dtype_override)

        return {
            "source_image": source_image,
            "target_image": target_image,
        }
