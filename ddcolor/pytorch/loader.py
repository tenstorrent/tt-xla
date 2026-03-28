# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DDColor model loader implementation for image colorization tasks.
"""

import torch
from torchvision import transforms
from transformers import AutoModel
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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available DDColor model variants."""

    DDCOLOR_MODELSCOPE = "modelscope"


class ModelLoader(ForgeModel):
    """DDColor model loader implementation for image colorization tasks."""

    _VARIANTS = {
        ModelVariant.DDCOLOR_MODELSCOPE: ModelConfig(
            pretrained_model_name="piddnad/ddcolor_modelscope",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DDCOLOR_MODELSCOPE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DDColor",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        input_tensor = transform(image).unsqueeze(0)

        if batch_size > 1:
            input_tensor = input_tensor.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            input_tensor = input_tensor.to(dtype_override)

        return input_tensor
