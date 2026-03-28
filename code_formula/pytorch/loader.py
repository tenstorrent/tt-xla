# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CodeFormula model loader implementation for code and formula OCR from images.
"""
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
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

IMAGE_SIZE = 1024
IMAGE_TOKEN_LEN = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelVariant(StrEnum):
    """Available CodeFormula model variants."""

    CODE_FORMULA = "CodeFormula"


class ModelLoader(ForgeModel):
    """CodeFormula model loader for code and formula OCR from images."""

    _VARIANTS = {
        ModelVariant.CODE_FORMULA: ModelConfig(
            pretrained_model_name="docling-project/CodeFormula",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CODE_FORMULA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CodeFormula",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(255, 255, 255))

        image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        pixel_values = image_transform(image).unsqueeze(0)

        # Build token sequence with image placeholders:
        # <img> + IMAGE_TOKEN_LEN * <imgpad> + </img> + prompt
        img_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
        img_end_token_id = self.tokenizer.convert_tokens_to_ids("</img>")
        imgpad_token_id = self.tokenizer.convert_tokens_to_ids("<imgpad>")

        prompt_text = "OCR: "
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        input_ids = (
            [img_token_id]
            + [imgpad_token_id] * IMAGE_TOKEN_LEN
            + [img_end_token_id]
            + prompt_ids
        )
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
