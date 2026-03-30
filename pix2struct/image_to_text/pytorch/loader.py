# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pix2Struct model loader implementation for image-to-text using PyTorch.
"""

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Pix2Struct PyTorch image-to-text model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Pix2Struct model loader implementation for image-to-text (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="google/pix2struct-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Pix2Struct",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import Pix2StructProcessor

        self._processor = Pix2StructProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import Pix2StructForConditionalGeneration

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Pix2StructForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from PIL import Image

        if self._processor is None:
            self._load_processor()

        image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        inputs = self._processor(
            images=image,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def decode_output(self, co_out):
        if self._processor is None:
            self._load_processor()

        generated_text = self._processor.batch_decode(co_out, skip_special_tokens=True)[
            0
        ]
        print(f"Generated text: {generated_text}")
        return generated_text
