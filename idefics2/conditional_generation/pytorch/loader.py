# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Idefics2 model loader implementation for multimodal conditional generation.
"""
import torch
from PIL import Image
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
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
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available Idefics2 model variants for conditional generation."""

    TINY_IDEFICS2 = "tiny_idefics2"


class ModelLoader(ForgeModel):
    """Idefics2 model loader implementation for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.TINY_IDEFICS2: LLMModelConfig(
            pretrained_model_name="trl-internal-testing/tiny-Idefics2ForConditionalGeneration",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_IDEFICS2

    sample_text = "What is shown in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="idefics2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Idefics2ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs

    def decode_output(self, co_out):
        if self.processor is None:
            self._load_processor()

        generated_text = self.processor.batch_decode(co_out, skip_special_tokens=True)[
            0
        ]
        print(f"Generated text: {generated_text}")
        return generated_text
