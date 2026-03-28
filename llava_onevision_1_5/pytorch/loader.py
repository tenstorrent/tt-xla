# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-OneVision-1.5 model loader implementation for multimodal conditional generation.
"""

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
    """Available LLaVA-OneVision-1.5 model variants."""

    LLAVA_ONEVISION_1_5_8B_INSTRUCT = "8B_Instruct"


class ModelLoader(ForgeModel):
    """LLaVA-OneVision-1.5 model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_ONEVISION_1_5_8B_INSTRUCT: ModelConfig(
            pretrained_model_name="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_ONEVISION_1_5_8B_INSTRUCT

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

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA-OneVision-1.5 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-OneVision-1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA-OneVision-1.5 model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = "auto"
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA-OneVision-1.5."""
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
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
