# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GOT-OCR 2.0 model loader implementation for image-to-text OCR tasks.
"""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GOT-OCR 2.0 model variants for image-to-text tasks."""

    GOT_OCR_2_0_HF = "got_ocr_2_0_hf"


class ModelLoader(ForgeModel):
    """GOT-OCR 2.0 model loader implementation for image-to-text OCR tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GOT_OCR_2_0_HF: LLMModelConfig(
            pretrained_model_name="stepfun-ai/GOT-OCR-2.0-hf",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GOT_OCR_2_0_HF

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="got_ocr_2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load Processor for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the processor's default dtype.

        Returns:
            The loaded processor instance
        """
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GOT-OCR 2.0 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GOT-OCR 2.0 model instance for image-to-text tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the GOT-OCR 2.0 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image = Image.open(
            __import__("requests")
            .get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg",
                stream=True,
            )
            .raw
        ).convert("RGB")

        inputs = self.processor(image, return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
