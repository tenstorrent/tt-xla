# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 VL for Video Classification model loader implementation.

muziyongshixin/Qwen2.5-VL-7B-for-VideoCls is a Qwen 2.5 VL model fine-tuned
for binary video classification. It uses the Qwen2_5_VLForConditionalGeneration
architecture with a classification head for single-label classification (2 labels).
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
    """Available Qwen 2.5 VL for Video Classification model variants."""

    QWEN_2_5_VL_7B_FOR_VIDEO_CLS = "7B_for_VideoCls"


class ModelLoader(ForgeModel):
    """Qwen 2.5 VL for Video Classification model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_2_5_VL_7B_FOR_VIDEO_CLS: LLMModelConfig(
            pretrained_model_name="muziyongshixin/Qwen2.5-VL-7B-for-VideoCls",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_VL_7B_FOR_VIDEO_CLS

    # Shared configuration parameters
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
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
            model="Qwen 2.5-VL for Video Classification",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Initialize processor with vision parameters
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 2.5 VL for Video Classification model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped model instance for video classification.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True, "use_cache": False}

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Apply chat template to get text prompt
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(self.messages)

        # Process all inputs together
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Convert pixel_values to specified dtype if provided
        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
