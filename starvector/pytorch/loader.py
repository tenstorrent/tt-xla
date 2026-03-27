# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
StarVector model loader implementation for image-to-SVG generation.
"""

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset

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
    """Available StarVector model variants."""

    STARVECTOR_1B = "1B"


class ModelLoader(ForgeModel):
    """StarVector model loader implementation for image-to-SVG generation tasks."""

    _VARIANTS = {
        ModelVariant.STARVECTOR_1B: ModelConfig(
            pretrained_model_name="starvector/starvector-1b-im2svg",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STARVECTOR_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="StarVector",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.processor = model.model.processor

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            pretrained_model_name = self._variant_config.pretrained_model_name
            self.processor = AutoProcessor.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"]

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
