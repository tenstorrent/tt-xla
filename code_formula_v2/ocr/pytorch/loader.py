# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CodeFormulaV2 model loader implementation for code and formula OCR tasks.

Uses Idefics3ForConditionalGeneration to recognize code snippets and
mathematical formulas from images.
"""
import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
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
    """Available CodeFormulaV2 model variants for OCR tasks."""

    CODE_FORMULA_V2 = "code_formula_v2"


class ModelLoader(ForgeModel):
    """CodeFormulaV2 model loader implementation for code and formula OCR tasks."""

    _VARIANTS = {
        ModelVariant.CODE_FORMULA_V2: LLMModelConfig(
            pretrained_model_name="docling-project/CodeFormulaV2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CODE_FORMULA_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="code_formula_v2",
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

        model = Idefics3ForConditionalGeneration.from_pretrained(
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
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
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

        return inputs

    def decode_output(self, co_out):
        if self.processor is None:
            self._load_processor()

        generated_text = self.processor.batch_decode(co_out, skip_special_tokens=True)[
            0
        ]
        print(f"Generated text: {generated_text}")
        return generated_text
