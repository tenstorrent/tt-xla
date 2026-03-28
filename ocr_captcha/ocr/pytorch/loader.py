# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OCR Captcha model loader implementation for optical character recognition of CAPTCHA images.
"""
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
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
    """Available OCR Captcha model variants."""

    OCR_CAPTCHA_V3 = "ocr-captcha-v3"


class ModelLoader(ForgeModel):
    """OCR Captcha model loader for CAPTCHA optical character recognition tasks."""

    _VARIANTS = {
        ModelVariant.OCR_CAPTCHA_V3: ModelConfig(
            pretrained_model_name="anuashok/ocr-captcha-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OCR_CAPTCHA_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ocr_captcha",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = TrOCRProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        return pixel_values

    @classmethod
    def decode_output(cls, outputs, processor=None, **kwargs):
        if processor is None:
            processor = TrOCRProcessor.from_pretrained("anuashok/ocr-captcha-v3")
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        return generated_text
