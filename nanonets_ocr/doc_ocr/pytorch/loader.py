# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nanonets-OCR-s model loader implementation for document OCR tasks.
"""
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
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
    """Available Nanonets-OCR model variants for document OCR tasks."""

    NANONETS_OCR_S = "nanonets_ocr_s"


class ModelLoader(ForgeModel):
    """Nanonets-OCR-s model loader implementation for document OCR tasks."""

    _VARIANTS = {
        ModelVariant.NANONETS_OCR_S: LLMModelConfig(
            pretrained_model_name="nanonets/Nanonets-OCR-s",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NANONETS_OCR_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="nanonets_ocr",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
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

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            attn_implementation="eager",
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        image = Image.open(
            __import__("requests").get(img_url, stream=True).raw
        ).convert("RGB")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_url},
                    {
                        "type": "text",
                        "text": "Extract the text from the above document as if you were reading it naturally.",
                    },
                ],
            },
        ]
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        ]
        inputs = self.processor(
            text=texts,
            images=image,
            padding=True,
            return_tensors="pt",
        )
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs
