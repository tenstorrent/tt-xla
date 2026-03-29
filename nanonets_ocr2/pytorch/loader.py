# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nanonets OCR2 model loader implementation for document OCR tasks.
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Optional


from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available Nanonets OCR2 model variants for document OCR tasks."""

    NANONETS_OCR2_1_5B_EXP = "1.5B_Exp"


class ModelLoader(ForgeModel):
    """Nanonets OCR2 model loader implementation for document OCR tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.NANONETS_OCR2_1_5B_EXP: LLMModelConfig(
            pretrained_model_name="nanonets/Nanonets-OCR2-1.5B-exp",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.NANONETS_OCR2_1_5B_EXP

    # Shared configuration parameters
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {
                    "type": "text",
                    "text": "Convert the document to markdown.",
                },
            ],
        }
    ]

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

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
            model="Nanonets OCR2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Nanonets OCR2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped Nanonets OCR2 model instance for document OCR tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Nanonets OCR2 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(self.messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
