# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PaddleOCR-VL-For-Manga model loader implementation for manga OCR tasks.
"""

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
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
    """Available PaddleOCR-VL-For-Manga model variants."""

    PADDLEOCR_VL_MANGA = "VL_For_Manga"


class ModelLoader(ForgeModel):
    """PaddleOCR-VL-For-Manga model loader for manga text recognition."""

    _VARIANTS = {
        ModelVariant.PADDLEOCR_VL_MANGA: ModelConfig(
            pretrained_model_name="jzhang533/PaddleOCR-VL-For-Manga",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PADDLEOCR_VL_MANGA

    sample_prompt = "Please read the text in this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PaddleOCR-VL-For-Manga",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PaddleOCR-VL-For-Manga model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
            "use_cache": False,
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the PaddleOCR-VL-For-Manga model."""
        if self.processor is None:
            self._load_processor()

        image = Image.new("RGB", (384, 384), color=(255, 255, 255))

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return dict(inputs)
